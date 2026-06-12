import json
import re
import shutil
import uuid
import warnings
from datetime import datetime
from functools import wraps
from itertools import batched
from pathlib import Path
from textwrap import dedent
from typing import Callable, Literal

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
from rich import print as pprint
from rich.progress import open as open_with_progress
from rich.progress import track
from rich.rule import Rule
from rich.status import Status

# suppress warnings for sentence-transformers
warnings.simplefilter(action="ignore", category=FutureWarning)


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


# helper to print a ruler before a function starts
def trace_start[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        pprint(Rule(f"{func.__name__}"))
        result = func(*args, **kwargs)
        return result

    return wrapper


# helper to print "ok" after a function finishes
def trace_finish[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        result = func(*args, **kwargs)
        print(f"{func.__name__} ok")
        return result

    return wrapper


# helper to delete all contents in a directory without deleting the directory itself like `shutil.rmtree`
def clean_dir(path: Path):
    if not path.exists() or not path.is_dir():
        return

    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)  # Delete subdirectories recursively


@trace_finish
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


@trace_finish
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


@trace_finish
def doc_get_chunks(markdown: str) -> list[str]:
    return CHUNKER(markdown, overlap=ENV.CHUNK_OVERLAP)  # type:ignore


@trace_finish
def doc_get_points(chunks: list[str], filename: str):
    points = list[PointStruct]()

    for batch_idx, batch in enumerate(
        batched(
            track(chunks, description="calculating embeddings"),
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


@trace_start
@trace_finish
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
        print("!! removing existing processed data since force is enabled\n---")
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
            # largest_idx_file_path is the path to the points file

            with (
                open_with_progress(
                    points_file_path,
                    "r",
                    description=f"loading points from {points_file_path}",
                ) as f,
            ):
                for lines in batched(f, ENV.BATCH_SIZE):
                    points = list[PointStruct]()
                    for line in lines:
                        record = Record.model_validate_json(line)
                        points.append(
                            PointStruct(
                                id=record.id,
                                vector=record.vector,  # type:ignore
                                payload=record.payload,
                            )
                        )
                    DB.upsert(collection_name=DB_COLLECTION_NAME, points=points)
            print("---")
        else:
            print(f"ingesting {raw_file_path.stem} ...")
            clean_dir(output_folder)

            html = doc_get_html(raw_file_path)
            (output_folder / f"01-{doc_get_html.__name__}.html").write_text(html)

            markdown = doc_get_markdown(html)
            (output_folder / f"02-{doc_get_markdown.__name__}.md").write_text(markdown)

            chunks = doc_get_chunks(markdown)
            (output_folder / f"03-{doc_get_chunks.__name__}.ndjson").write_text(
                "\n".join(json.dumps(chunk) for chunk in chunks)
            )

            points = doc_get_points(chunks, raw_file_path.stem)
            (output_folder / f"04-{doc_get_points.__name__}.ndjson").write_text(
                "\n".join(point.model_dump_json() for point in points)
            )

            DB.upload_points(collection_name=DB_COLLECTION_NAME, points=points)
            print("---")

    print(f"db initialized with {DB.count(DB_COLLECTION_NAME, exact=True).count} points")


@trace_finish
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


@trace_finish
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


@trace_finish
def query_get_llm_res(prompt: str):
    if ENV.LLM_MODEL_NAME is None:
        return ""

    if ENV.LLM_MODEL_NAME not in {m["model"] for m in ollama.list()["models"]}:
        print(f"model missing from ollama, pulling {ENV.LLM_MODEL_NAME} ...")
        ollama.pull(ENV.LLM_MODEL_NAME)

    with Status("collecting ollama outputs ..."):
        return "".join(
            res["message"]["content"]
            for res in ollama.chat(  # type:ignore
                model=ENV.LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": LLM_SYSTEM_INSTRUCTION,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options=ollama.Options(
                    num_ctx=ENV.LLM_NUM_CTX,
                    temperature=ENV.LLM_TEMPERATURE,
                ),
                think=ENV.LLM_THINKING,
                stream=True,
            )
        )


@trace_start
@trace_finish
def handle_query(query: str):
    pprint(f"{query=}")

    output_folder = ENV.PATH_TO_HISTORY_FOLDER / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "01-environment.json").write_text(ENV.model_dump_json(indent=2))

    retrieved_points = query_get_points(query)
    (output_folder / "02-retrieved_context.json").write_text(
        "\n".join([point.model_dump_json() for point in retrieved_points])
    )

    prompt = query_get_prompt(query, retrieved_points)
    (output_folder / "03-prompt.xml").write_text(prompt)

    llm_res = query_get_llm_res(prompt)
    (output_folder / "04-response.md").write_text(llm_res)


# region main

pprint(f"{ENV=}\n")
pprint(f"{ARGS=}\n")

init_db()

# TODO: from rich.prompt import Prompt, then Prompt.ask("Enter your question")
QUERY = "What is the business model of Nvidia?"
handle_query(QUERY)

# endregion
