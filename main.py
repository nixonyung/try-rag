import inspect
import json
import re
import shutil
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime
from itertools import batched
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Literal

import chardet
import html_to_markdown
import mdformat
import numpy as np
import ollama
import semchunk
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from lxml import html as lxml_html
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    Prefetch,
    Record,
    Rrf,
    RrfQuery,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm
from wcwidth import wcswidth

# suppress warnings for sentence-transformers
warnings.simplefilter(action="ignore", category=FutureWarning)

# region configs


class Environment(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    PATH_TO_RAW_DATA_FOLDER: Path
    PATH_TO_PROCESSED_DATA_FOLDER: Path
    PATH_TO_HISTORY_FOLDER: Path
    PATH_TO_SYSTEM_INSTRUCTION: Path

    CHUNKER_MODEL_NAME: str | None = None
    CHUNK_SIZE: int
    CHUNK_OVERLAP: float

    DENSE_EMB_MODEL_CONFIG: tuple[Literal["fastembed"] | None, str] = (None, "")
    SPARSE_EMB_MODEL_CONFIG: tuple[Literal["fastembed"] | None, str] = (None, "")
    BATCH_SIZE: int

    RETRIEVAL_SCORE_THRESHOLD: float
    RETRIEVAL_LIMIT: int

    LLM_MODEL_NAME: str | None = None
    LLM_TEMPERATURE: float
    LLM_THINKING: bool
    LLM_NUM_CTX: int


ENV = Environment()  # type:ignore


class Arguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    FORCE: CliImplicitFlag[bool] = False


ARGS = Arguments()  # type:ignore

# endregion

# region logging helpers

log_section_depth = 0
LOG_INDENT_SIZE = 4
CWD = Path.cwd()


def log_with_lineno(message: str, frame_offset: int = 2):
    frame = inspect.stack()[frame_offset]
    filepath = Path(frame.filename).relative_to(CWD)
    lineno = frame.lineno

    # the total width from the end of message to the end of terminal
    padding_width = shutil.get_terminal_size().columns - wcswidth(message)

    print(f"{message}{f'({filepath}:{lineno}) ':>{padding_width}}")


def log_warning(message: str):
    log_with_lineno(
        f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'!! ':<{LOG_INDENT_SIZE}}{message}"
    )


def log_finished(message: str):
    log_with_lineno(
        f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'✔ ':<{LOG_INDENT_SIZE}}{message}"
    )


def log_header(message: str):
    print()
    log_with_lineno(f"{'=' * (LOG_INDENT_SIZE - 1)} {message} {'=' * (LOG_INDENT_SIZE - 1)}")


def log_progress[T](iterable: Iterable[T], desc: str):
    frame = inspect.stack()[2]
    filepath = Path(frame.filename).relative_to(CWD)
    lineno = frame.lineno

    return tqdm(
        iterable,
        desc=f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'• ':<{LOG_INDENT_SIZE}}{desc}",
        postfix=f"{filepath}:{lineno}",
    )


@contextmanager
def log_section(title: str):
    global log_section_depth

    try:
        log_with_lineno(
            f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'▶ ':<{LOG_INDENT_SIZE}}{title}",
            frame_offset=3,
        )
        log_section_depth += 1
        yield
    finally:
        log_section_depth -= 1


# endregion

# region fs helpers


# helper to delete all contents in a directory without deleting the directory itself like `shutil.rmtree`
def clean_dir(path: Path):
    if not path.exists() or not path.is_dir():
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def load_lines_with_progress(path: Path, desc: str):
    frame = inspect.stack()[2]
    filepath = Path(frame.filename).relative_to(CWD)
    lineno = frame.lineno

    with (
        path.open() as f,
        tqdm(
            desc=f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'• ':<{LOG_INDENT_SIZE}}{desc}",
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            postfix=f"{filepath}:{lineno}",
        ) as pbar,
    ):
        for line in f:
            pbar.update(len(line.encode()))
            yield line


# endregion

# region ollama helpers


def ollama_generate(
    model: str,
    system_instruction: str,
    prompt: str,
    temperature: float = ENV.LLM_TEMPERATURE,
    num_ctx: int = ENV.LLM_NUM_CTX,
    thinking: bool = ENV.LLM_THINKING,
):
    if model not in {m["model"] for m in ollama.list()["models"]}:
        log_warning(f"model missing from ollama, pulling {model} ...")
        ollama.pull(model=model)

    return "".join(
        res["message"]["content"]
        for res in log_progress(
            ollama.chat(  # type:ignore
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options=ollama.Options(
                    temperature=temperature,
                    num_ctx=num_ctx,
                ),
                think=thinking,
                stream=True,
            ),
            desc="collecting ollama outputs",
        )
    )


# endregion

# region context

DB = QdrantClient(":memory:")
DB_COLLECTION_NAME = "documents"

CHUNKER = semchunk.chunkerify(  # type:ignore
    ENV.CHUNKER_MODEL_NAME
    if ENV.CHUNKER_MODEL_NAME is not None
    else lambda text: len(text.split()),
    chunk_size=ENV.CHUNK_SIZE,
)

DENSE_EMB_MODEL = (
    TextEmbedding(ENV.DENSE_EMB_MODEL_CONFIG[1])
    if ENV.DENSE_EMB_MODEL_CONFIG[0] == "fastembed"
    else None
)
DENSE_EMB_FALLBACK_DIMENSION = 128
SPARSE_EMB_MODEL = (
    SparseTextEmbedding(ENV.SPARSE_EMB_MODEL_CONFIG[1])
    if ENV.SPARSE_EMB_MODEL_CONFIG[0] == "fastembed"
    else None
)
# DENSE_EMBEDDINGS_DIMENSION = 512
# DENSE_MODEL = SentenceTransformer(
#     "jinaai/jina-embeddings-v5-text-nano",
#     trust_remote_code=True,
#     model_kwargs={"dtype": torch.bfloat16},
# )
# DENSE_EMBEDDINGS_DIMENSION = 768
# DENSE_MODEL = distill("jinaai/jina-embeddings-v5-text-nano", trust_remote_code=True)
# DENSE_EMBEDDINGS_DIMENSION = 256
# SPARSE_MODEL = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
# SPARSE_MODEL = SparseEncoder("prithivida/Splade_PP_en_v2")

LLM_SYSTEM_INSTRUCTION = dedent(ENV.PATH_TO_SYSTEM_INSTRUCTION.read_text())

# endregion


def doc_get_html(raw_file_path: Path):
    root = lxml_html.parse(
        raw_file_path,
        parser=lxml_html.HTMLParser(
            encoding=chardet.detect(raw_file_path.read_bytes())["encoding"],
            remove_blank_text=True,
            remove_comments=True,
        ),
    )

    for element in list(root.iter()):
        # filter out invisible nodes
        if element.tag in {"script", "noscript", "style", "img"} or any(
            s in element.attrib.get("style", "").replace(" ", "")
            for s in ["display:none", "visibility:hidden"]
        ):
            if (parent := element.getparent()) is not None:
                parent.remove(element)
            continue

        # remove links by transforming <a> to <span>
        if element.tag == "a":
            element.attrib.pop("href", None)
            element.tag = "span"

        # filter out unwanted attributes
        for k in element.attrib.keys():
            if k not in {"style", "colspan", "rowspan"}:
                element.attrib.pop(k)

    return lxml_html.tostring(root, encoding="unicode", pretty_print=True)


def doc_get_markdown(html: str):
    markdown = (
        html_to_markdown.convert(
            html,
            options=html_to_markdown.ConversionOptions(extract_metadata=False),
        ).content
        or ""
    )

    # character-level cleaning
    markdown = markdown.translate(
        {
            ord(" "): " ",
            ord("’"): "'",
            ord("“"): '"',
            ord("”"): '"',
            ord("•"): "\n- ",
        }
    )
    markdown = re.sub(r"----+", "---", markdown)

    markdown = mdformat.text(markdown)  # type:ignore

    return markdown


def doc_get_chunks(markdown: str) -> list[str]:
    return CHUNKER(markdown, overlap=ENV.CHUNK_OVERLAP)  # type:ignore


def doc_get_points(chunks: list[str], filename: str):
    points = list[PointStruct]()

    for batch_idx, batch in enumerate(
        batched(
            log_progress(chunks, desc="calculating embeddings"),
            ENV.BATCH_SIZE,
        )
    ):
        dense_embs = (
            list(DENSE_EMB_MODEL.passage_embed(batch))
            if DENSE_EMB_MODEL is not None
            else [np.zeros(DENSE_EMB_FALLBACK_DIMENSION)] * len(batch)
        )
        sparse_embs = (
            list(SPARSE_EMB_MODEL.passage_embed(batch))
            if SPARSE_EMB_MODEL is not None
            else [SparseEmbedding(indices=np.empty(0, dtype=np.int32), values=np.empty(0))]
            * len(batch)
        )

        # dense_emds = DENSE_MODEL.encode(  # type:ignore
        #     batch,
        #     task="retrieval",
        #     prompt_name="document",
        # )
        # sparse_emds = SPARSE_MODEL.encode(batch)  # type:ignore

        points.extend(
            PointStruct(
                id=uuid.uuid7(),
                vector={
                    "dense": dense_embs[i].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_embs[i].indices.tolist(),
                        values=sparse_embs[i].values.tolist(),
                    ),
                    # "dense": dense_emds[i].tolist(),
                    # "sparse": SparseVector(
                    #     indices=sparse_emds[i].coalesce().indices()[0].tolist(),
                    #     values=sparse_emds[i].coalesce().values().tolist(),
                    # ),
                },
                payload={
                    "filename": filename,
                    "chunk_idx": batch_idx * ENV.BATCH_SIZE + i,
                    "document": chunk,
                },
            )
            for i, chunk in enumerate(batch)
        )

    return points


def init_db():
    DB.create_collection(
        DB_COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_EMB_MODEL.embedding_size
                if DENSE_EMB_MODEL is not None
                else DENSE_EMB_FALLBACK_DIMENSION,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    if ENV.PATH_TO_PROCESSED_DATA_FOLDER.exists() and ARGS.FORCE:
        log_warning("removing existing processed data since force is enabled")
        clean_dir(ENV.PATH_TO_PROCESSED_DATA_FOLDER)
    for raw_file_path in ENV.PATH_TO_RAW_DATA_FOLDER.glob("*.html"):
        output_folder = ENV.PATH_TO_PROCESSED_DATA_FOLDER / raw_file_path.stem
        output_folder.mkdir(parents=True, exist_ok=True)

        # load existing points if the file exists and is valid
        # (i.e. the file has the largest index and the file name ends with the correct function name)
        points_file_path = max(
            [f for f in output_folder.iterdir() if f.is_file()],
            key=lambda f: f.stem,
            default=None,
        )
        if points_file_path is not None and points_file_path.stem.endswith(doc_get_points.__name__):
            for lines in batched(
                load_lines_with_progress(
                    points_file_path,
                    desc=f"loading points from {points_file_path}",
                ),
                ENV.BATCH_SIZE,
            ):
                DB.upsert(
                    collection_name=DB_COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=record.id,
                            vector=record.vector,  # type:ignore
                            payload=record.payload,
                        )
                        for line in lines
                        if (record := Record.model_validate_json(line))
                    ],
                )
        else:
            with log_section(f"ingesting document {raw_file_path} ..."):
                clean_dir(output_folder)

                html = doc_get_html(raw_file_path)
                (output_folder / f"01-{doc_get_html.__name__}.html").write_text(html)
                log_finished(doc_get_html.__name__)

                markdown = doc_get_markdown(html)
                (output_folder / f"02-{doc_get_markdown.__name__}.md").write_text(markdown)
                log_finished(doc_get_markdown.__name__)

                chunks = doc_get_chunks(markdown)
                (output_folder / f"03-{doc_get_chunks.__name__}.ndjson").write_text(
                    "\n".join(json.dumps(chunk) for chunk in chunks)
                )
                log_finished(doc_get_chunks.__name__)

                points = doc_get_points(chunks, raw_file_path.stem)
                (output_folder / f"04-{doc_get_points.__name__}.ndjson").write_text(
                    "\n".join(point.model_dump_json() for point in points)
                )
                log_finished(doc_get_points.__name__)

                DB.upload_points(collection_name=DB_COLLECTION_NAME, points=points)

    log_finished(f"db initialized with {DB.count(DB_COLLECTION_NAME, exact=True).count} points")


def query_get_points(query: str):
    if DENSE_EMB_MODEL is None or SPARSE_EMB_MODEL is None:
        return list[ScoredPoint]()

    dense_emb = list(DENSE_EMB_MODEL.query_embed(query))[0]
    sparse_emb = list(SPARSE_EMB_MODEL.query_embed(query))[0]

    return DB.query_points(
        DB_COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_emb.tolist(),
                using="dense",
                score_threshold=ENV.RETRIEVAL_SCORE_THRESHOLD,
                limit=ENV.RETRIEVAL_LIMIT,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
                using="sparse",
                score_threshold=ENV.RETRIEVAL_SCORE_THRESHOLD,
                limit=ENV.RETRIEVAL_LIMIT,
            ),
        ],
        query=RrfQuery(rrf=Rrf()),
    ).points


def query_get_prompt(query: str, retrieved_points: list[ScoredPoint]):
    parts = list[str]()
    parts.append("<retrieved_context>")
    for point in retrieved_points:
        if point.payload is None:
            continue

        parts.append("<document_chunk>")
        parts.append("<metadata>")
        for k, v in point.payload.items():
            if k == "document":
                continue
            parts.append(f"{k}: {v}")
        parts.append("</metadata>")
        parts.append("<content>")
        parts.append(point.payload["document"])
        parts.append("</content>")
        parts.append("</document_chunk>")
    parts.append("</retrieved_context>")

    parts.append("<user_question>")
    parts.append(query)
    parts.append("</user_question>")

    return "\n".join(parts)


def query_get_llm_res(prompt: str):
    if ENV.LLM_MODEL_NAME is None:
        return ""

    return ollama_generate(
        ENV.LLM_MODEL_NAME,
        LLM_SYSTEM_INSTRUCTION,
        prompt,
    )


def handle_query(query: str):
    print(f"{query=}")

    output_folder = ENV.PATH_TO_HISTORY_FOLDER / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "01-environment.json").write_text(ENV.model_dump_json(indent=2))

    retrieved_points = query_get_points(query)
    (output_folder / "02-retrieved_context.json").write_text(
        "\n".join([point.model_dump_json() for point in retrieved_points])
    )
    log_finished(query_get_points.__name__)

    prompt = query_get_prompt(query, retrieved_points)
    (output_folder / "03-prompt.xml").write_text(prompt)
    log_finished(query_get_prompt.__name__)

    llm_res = query_get_llm_res(prompt)
    (output_folder / "04-response.md").write_text(llm_res)
    log_finished(query_get_llm_res.__name__)


# region main

print(f"\nENV = {ENV.model_dump_json(indent=4)}")
print(f"\nARGS = {ARGS.model_dump_json(indent=4)}")

log_header(f"{init_db.__name__}")
init_db()

# TODO: from rich.prompt import Prompt, then Prompt.ask("Enter your question")
QUERY = "What is the business model of Nvidia?"

log_header(f"{handle_query.__name__}")
handle_query(QUERY)

# endregion
