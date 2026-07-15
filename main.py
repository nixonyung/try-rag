# pyright: reportUnknownMemberType=false

import inspect
import re
import shutil
import uuid
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from itertools import batched
from pathlib import Path
from textwrap import dedent
from typing import Literal

import chardet
import html_to_markdown
import mdformat
import numpy as np
import ollama
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from fastembed.common.types import NumpyArray
from lxml import html as lxml_html
from lxml.etree import _ElementTree  # pyright: ignore[reportPrivateUsage]
from lxml.html import HtmlElement
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
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
from semchunk import semchunk
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

    CHUNKER_MODEL_NAME: str | None = None
    CHUNK_SIZE: int
    CHUNK_OVERLAP: float

    DENSE_EMB_MODEL_CONFIG: tuple[Literal["fastembed"] | None, str] = (None, "")
    SPARSE_EMB_MODEL_CONFIG: tuple[Literal["fastembed"] | None, str] = (None, "")
    BATCH_SIZE: int

    RETRIEVAL_SCORE_THRESHOLD: float
    RETRIEVAL_LIMIT: int

    LLM_CHUNKING_MODEL_NAME: str | None = None
    LLM_QUERY_MODEL_NAME: str | None = None
    LLM_CHUNKING_INSTRUCTION_PATH: Path
    LLM_QUERY_INSTRUCTION_PATH: Path
    LLM_TEMPERATURE: float
    LLM_THINKING: bool
    LLM_DEFAULT_NUM_CTX: int


ENV = Environment()  # pyright: ignore[reportCallIssue]


class Arguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    FORCE: CliImplicitFlag[bool] = False


ARGS = Arguments()

# endregion

# region logging helpers

LOG_INDENT_SIZE = 4
CWD = Path.cwd()

log_section_depth = 0


def _log_with_lineno(message: str, frame_offset: int):
    frame = inspect.stack()[2 + frame_offset]
    filepath = Path(frame.filename).relative_to(CWD)
    lineno = frame.lineno

    # the total width from the end of message to the end of terminal
    padding_width = max(0, shutil.get_terminal_size().columns - wcswidth(message))

    print(f"{message}{f'({filepath}:{lineno}) ':>{padding_width}}")


def log_warning(message: str, frame_offset: int = 0):
    _log_with_lineno(
        f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'!! ':<{LOG_INDENT_SIZE}}{message}",
        frame_offset,
    )


def log_finished(message: str, frame_offset: int = 0):
    _log_with_lineno(
        f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'✔ ':<{LOG_INDENT_SIZE}}{message}",
        frame_offset,
    )


@contextmanager
def log_header(message: str, frame_offset: int = 0):
    banner_size = max(LOG_INDENT_SIZE, shutil.get_terminal_size().columns // 4 - len(message) // 2)

    print()
    _log_with_lineno(
        f"{'=' * banner_size} {message} {'=' * banner_size}",
        1 + frame_offset,
    )

    try:
        yield
    finally:
        pass


def log_progress[T](iterable: Iterable[T], desc: str, frame_offset: int = 0):
    frame = inspect.stack()[2 + frame_offset]
    filepath = Path(frame.filename).relative_to(CWD)
    lineno = frame.lineno

    return tqdm(
        iterable,
        desc=f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'• ':<{LOG_INDENT_SIZE}}{desc}",
        postfix=f"{filepath}:{lineno}",
    )


@contextmanager
def log_section(title: str, frame_offset: int = 0):
    global log_section_depth

    _log_with_lineno(
        f"{' ' * LOG_INDENT_SIZE * log_section_depth}{'▶ ':<{LOG_INDENT_SIZE}}{title}",
        frame_offset=1 + frame_offset,
    )
    log_section_depth += 1

    try:
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


def read_html(path: Path):
    return lxml_html.parse(
        path,
        parser=lxml_html.HTMLParser(
            encoding=chardet.detect(path.read_bytes())["encoding"],
            remove_blank_text=True,
            remove_comments=True,
        ),
    )


def read_lines_with_progress(path: Path, desc: str):
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


def write_to_file(folder: Path, filename: str, data: str):
    if folder.exists() and not folder.is_dir():
        return

    folder.mkdir(parents=True, exist_ok=True)

    num_files = sum(1 for item in folder.iterdir() if item.is_file())
    path = folder / f"{num_files + 1:02}-{filename}"

    path.write_text(data)
    log_finished(
        f"wrote to {filename}{' ' * LOG_INDENT_SIZE}({path})",
        frame_offset=1,
    )


# endregion

# region ollama helpers


def ollama_generate(
    model: str | None,
    system_instruction: str,
    prompt: str,
    format: type[BaseModel] | None = None,
    temperature: float = ENV.LLM_TEMPERATURE,
    num_ctx: int = ENV.LLM_DEFAULT_NUM_CTX,
    thinking: bool = ENV.LLM_THINKING,
    desc: str | None = None,
):
    if model is None:
        return ""
    model = model.strip()

    if model not in {m["model"] for m in ollama.list()["models"]}:
        log_warning(f"model missing from ollama, pulling {model} ...")
        ollama.pull(model=model)

    stream = ollama.chat(
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
        format=format.model_json_schema() if format is not None else None,
        options=ollama.Options(
            temperature=temperature,
            num_ctx=num_ctx,
        ),
        think=thinking,
        stream=True,
    )
    if desc is not None:
        stream = log_progress(stream, desc=desc)

    return "".join(res.message.content or "" for res in stream)


# endregion

# region context

DB = QdrantClient(":memory:")
DB_COLLECTION_NAME = "documents"

CHUNKER = semchunk.chunkerify(
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

# endregion

# region data cleaning procedures

HTML_TRANSLATE_MAP = {
    # Spaces
    ord(" "): " ",  # Non-breaking space (NBSP)
    ord("\u200b"): "",  # Zero-width space (remove completely)
    ord("\u2002"): " ",  # En space
    ord("\u2003"): " ",  # Em space
    # Quotes
    ord("’"): "'",  # Right single quote / apostrophe
    ord("‘"): "'",  # Left single quote
    ord("“"): '"',  # Left double quote
    ord("”"): '"',  # Right double quote
    ord("„"): '"',  # Low-double prime quote
    # Dashes
    ord("–"): "-",  # En dash
    ord("—"): "-",  # Em dash
    # Miscellaneous
    ord("…"): "...",  # Ellipsis
    ord("˜"): "~",  # Small tilde
}


def clean_html(root: _ElementTree[HtmlElement]):
    # caveat 1: changes to text content and attributes will be overrided by structual changes
    #
    # caveat 2: structual changes should not be performed inplace during root.iter(),
    # have to extract "snapshots" using root.xpath() first

    # filter out invisible nodes
    for element in root.xpath(
        "//script"
        "| //noscript"
        "| //style"
        "| //img"
        "| //*[contains(normalize-space(@style), 'display:none')]"
        "| //*[contains(normalize-space(@style), 'visibility:hidden')]"
    ):
        if (parent := element.getparent()) is not None:
            parent.remove(element)

    # strip ix tags but keep the content
    for element in root.xpath("//*[starts-with(local-name(), 'ix')]"):
        element.drop_tag()

    # perform inplace changes
    for element in root.iter():
        # transform <a> to <span>
        if element.tag == "a":
            element.attrib.pop("href", None)
            element.tag = "span"

        # character-level cleaning in text contents
        if element.text:
            element.text = element.text.translate(HTML_TRANSLATE_MAP)

        # filter out unwanted attributes
        for k in element.attrib.keys():
            if k not in {"colspan", "rowspan"}:
                element.attrib.pop(k)


def to_markdown(html: str):
    markdown = (
        html_to_markdown.convert(
            html,
            options=html_to_markdown.ConversionOptions(extract_metadata=False),
        ).content
        or ""
    )
    markdown = mdformat.text(markdown)

    return markdown


def clean_markdown(markdown: str):
    # remove footer
    markdown = re.sub(
        r"\n\d{1,2}\n(\n______________________________________________________________________\n)?",
        "",
        markdown,
    )
    markdown = re.sub(r"Table of Contents\n\n(?!\|)", "", markdown)

    markdown = markdown.replace("•", "- ")

    markdown = mdformat.text(markdown)

    return markdown


# endregion

# region chunking procedures


def to_chunks(markdown: str) -> list[str]:
    return CHUNKER(markdown, overlap=ENV.CHUNK_OVERLAP)  # pyright: ignore[reportReturnType]


def gen_db_points(chunks: list[str], filename: str):
    points = list[PointStruct]()

    for batch_idx, batch in enumerate(
        batched(
            log_progress(chunks, desc="calculating embeddings"),
            ENV.BATCH_SIZE,
        )
    ):
        dense_embs: list[NumpyArray]
        match ENV.DENSE_EMB_MODEL_CONFIG:
            case ("fastembed", _) if DENSE_EMB_MODEL is not None:
                dense_embs = list(DENSE_EMB_MODEL.passage_embed(batch))
            case _:
                dense_embs = [np.zeros(DENSE_EMB_FALLBACK_DIMENSION)] * len(batch)

        sparse_embs: list[SparseEmbedding]
        match ENV.SPARSE_EMB_MODEL_CONFIG:
            case ("fastembed", _) if SPARSE_EMB_MODEL is not None:
                sparse_embs = list(SPARSE_EMB_MODEL.passage_embed(batch))
            case _:
                sparse_embs = [
                    SparseEmbedding(indices=np.empty(0, dtype=np.int32), values=np.empty(0))
                ] * len(batch)

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


class Chunk(BaseModel):
    class ChunkResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")

        atomic_sentences: list[str] = Field(
            description=(
                "A list of factual, split, self-contained plain-text sentences derived EXCLUSIVELY from <input_text>. "
                "Include ONLY material financials, business model mechanics, supply chain vulnerabilities, "
                "market research, corporate governance, key human resources talent/incentives, and structural risks. "
                "Ignore vague corporate platitudes, boilerplate legal definitions, and sentences found within <last_processed_sentences>."
            ),
        )

    response: ChunkResponse

    filestem: str
    line_start_idx: int
    line_end_idx: int
    passage: str


CHUNKS = dict[str, list[Chunk]]()  # key: filestem, value: list of chunks

LLM_CHUNKING_INSTRUCTION = dedent(ENV.LLM_CHUNKING_INSTRUCTION_PATH.read_text())


def to_chunks_v2(markdown: str, filestem: str):
    chunks = list[Chunk]()

    # debug
    Path("testing_output").mkdir(parents=True, exist_ok=True)

    lines = list[str]()
    line_start_idx = 0
    word_count = 0
    for i, line in enumerate(log_progress(markdown.splitlines(), desc="chunking markdown")):
        lines.append(line)
        word_count += len(line.split())

        if word_count >= ENV.CHUNK_SIZE:
            parts = list[str]()

            parts.append(
                "<!-- PRONOUN REFERENCE POOL: READ-ONLY. DO NOT EXTRACT OR RE-PRODUCE THESE SENTENCES. -->"
            )
            parts.append("<last_processed_sentences>")
            parts.extend(
                f"<sentence>{sentence}</sentence>"
                for sentence in (chunks[-1].response.atomic_sentences[-3:] if chunks else [])
            )
            parts.append("</last_processed_sentences>")
            parts.append("")
            parts.append("<!-- TARGET TEXT: EXTRACT FROM THIS SECTION ONLY -->")
            parts.append("<input_text>")
            parts.extend(lines)
            parts.append("</input_text>")

            prompt = "\n".join(line for line in parts)
            # debug
            Path(
                f"testing_output/{i:03}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
            ).write_text(prompt)

            if ollama_res := ollama_generate(
                ENV.LLM_CHUNKING_MODEL_NAME,
                system_instruction=LLM_CHUNKING_INSTRUCTION,
                prompt=prompt,
                format=Chunk.ChunkResponse,
                num_ctx=8192,
            ):
                chunk_res = Chunk.ChunkResponse.model_validate_json(ollama_res)
                chunk = Chunk(
                    response=chunk_res,
                    filestem=filestem,
                    line_start_idx=line_start_idx,
                    line_end_idx=i,
                    passage="\n".join(lines),
                )
                # debug
                Path(
                    f"testing_output/{i:03}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
                ).write_text(chunk.model_dump_json(indent=2))
                chunks.append(chunk)

            lines.clear()
            line_start_idx = i + 1
            word_count = 0

    return chunks


class DBPointPayload(BaseModel):
    atomic_sentence: str
    filestem: str
    chunk_idx: int


def gen_db_points_v2(chunks: list[Chunk]):
    points = list[PointStruct]()

    for batch in batched(
        log_progress(
            (
                DBPointPayload(
                    atomic_sentence=sentence,
                    filestem=chunk.filestem,
                    chunk_idx=i,
                )
                for i, chunk in enumerate(chunks)
                for sentence in chunk.response.atomic_sentences
            ),
            desc="calculating embeddings",
        ),
        ENV.BATCH_SIZE,
    ):
        dense_embs: list[NumpyArray]
        match ENV.DENSE_EMB_MODEL_CONFIG:
            case ("fastembed", _) if DENSE_EMB_MODEL is not None:
                dense_embs = list(
                    DENSE_EMB_MODEL.passage_embed([payload.atomic_sentence for payload in batch])
                )
            case _:
                dense_embs = [np.zeros(DENSE_EMB_FALLBACK_DIMENSION)] * len(batch)

        sparse_embs: list[SparseEmbedding]
        match ENV.SPARSE_EMB_MODEL_CONFIG:
            case ("fastembed", _) if SPARSE_EMB_MODEL is not None:
                sparse_embs = list(
                    SPARSE_EMB_MODEL.passage_embed([payload.atomic_sentence for payload in batch])
                )
            case _:
                sparse_embs = [
                    SparseEmbedding(indices=np.empty(0, dtype=np.int32), values=np.empty(0))
                ] * len(batch)

        points.extend(
            PointStruct(
                id=uuid.uuid7(),
                vector={
                    "dense": dense_embs[i].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_embs[i].indices.tolist(),
                        values=sparse_embs[i].values.tolist(),
                    ),
                },
                payload=payload.model_dump(),
            )
            for i, payload in enumerate(batch)
        )

    return points


# endregion

# region query procedures


def db_query(query: str):
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


def get_prompt_for_query(query: str, retrieved_points: list[ScoredPoint]):
    return dedent(f"""\
        <retrieved_context>
        {
        "\n\n".join(
            dedent(f'''\
                <document_chunk>
                <metadata>
                {"\n".join(f"{k}: {v}" for k, v in point.payload.items() if k != "document")}
                </metadata>
                <content>
                {point.payload["document"]}
                </content>
                </document_chunk>
            ''')
            for point in retrieved_points
            if point.payload is not None
        )
    }
        </retrieved_context>

        <user_question>
        {query}
        </user_question>
    """)


def get_prompt_for_query_v2(query: str, retrieved_points: list[ScoredPoint]):
    parts = list[str]()

    parts.append("<retrieved_documents>")
    for point in retrieved_points:
        if point.payload is None:
            continue

        payload = DBPointPayload.model_validate(point.payload)

        parts.append("<document>")

        parts.append("<metadata>")
        parts.append(f"filename: {payload.filestem}")
        parts.append(f"chunk_idx: {payload.chunk_idx}")
        parts.append("</metadata>")

        parts.append("<matching_facts>")
        parts.extend(
            f"<fact>{sentence}</fact>"
            for sentence in CHUNKS[payload.filestem][payload.chunk_idx].response.atomic_sentences
        )
        parts.append("</matching_facts>")

        parts.append("<content>")
        parts.append(CHUNKS[payload.filestem][payload.chunk_idx].passage)
        parts.append("</content>")

        parts.append("</document>")
    parts.append("</retrieved_documents>")

    parts.append("<user_query>")
    parts.append(query)
    parts.append("</user_query>")

    return "\n".join(parts)


# endregion

if __name__ == "__main__":
    print(f"\nENV = {ENV.model_dump_json(indent=4)}")
    print(f"\nARGS = {ARGS.model_dump_json(indent=4)}")

    POINTS_FILENAME = "points.ndjson"

    with log_header("init DB"):
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
            if points_file_path is not None and points_file_path.name.endswith(POINTS_FILENAME):
                for lines in batched(
                    read_lines_with_progress(
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
                                vector=record.vector,  # pyright: ignore[reportArgumentType]
                                payload=record.payload,
                            )
                            for line in lines
                            if (record := Record.model_validate_json(line))
                        ],
                    )
            else:
                with log_section(f"ingesting document {raw_file_path} ..."):
                    clean_dir(output_folder)

                    root = read_html(raw_file_path)
                    write_to_file(
                        output_folder,
                        "original-formatted.html",
                        lxml_html.tostring(root, encoding="unicode", pretty_print=True),
                    )

                    clean_html(root)
                    html = lxml_html.tostring(root, encoding="unicode", pretty_print=True)
                    write_to_file(output_folder, "cleaned.html", html)

                    markdown = to_markdown(html)
                    write_to_file(output_folder, "converted.md", markdown)

                    markdown = clean_markdown(markdown)
                    write_to_file(output_folder, "converted-cleaned.md", markdown)

                    # chunks = to_chunks(markdown)
                    # write_to_file(
                    #     output_folder,
                    #     "chunked.ndjson",
                    #     "\n".join(json.dumps(chunk) for chunk in chunks),
                    # )

                    # points = gen_db_points(chunks, raw_file_path.stem)
                    # write_to_file(
                    #     output_folder,
                    #     POINTS_FILENAME,
                    #     "\n".join(point.model_dump_json() for point in points),
                    # )

                    chunks = to_chunks_v2(markdown, raw_file_path.stem)
                    write_to_file(
                        output_folder,
                        "chunks.json",
                        TypeAdapter(list[Chunk]).dump_json(chunks, indent=2).decode(),
                    )
                    CHUNKS[raw_file_path.stem] = chunks

                    points = gen_db_points_v2(chunks)
                    write_to_file(
                        output_folder,
                        POINTS_FILENAME,
                        "\n".join(point.model_dump_json() for point in points),
                    )

                    DB.upload_points(collection_name=DB_COLLECTION_NAME, points=points)

        log_finished(f"DB initialized with {DB.count(DB_COLLECTION_NAME, exact=True).count} points")

    # TODO: from rich.prompt import Prompt, then Prompt.ask("Enter your question")
    query = "What is the business model of Nvidia?"

    LLM_QUERY_INSTRUCTION = dedent(ENV.LLM_QUERY_INSTRUCTION_PATH.read_text())

    with log_header("handle query"):
        print(f"{query=}")

        output_folder = ENV.PATH_TO_HISTORY_FOLDER / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_folder.mkdir(parents=True, exist_ok=True)
        write_to_file(
            output_folder,
            "environment.json",
            ENV.model_dump_json(indent=2),
        )

        retrieved_points = db_query(query)
        write_to_file(
            output_folder,
            "retrieved_context.json",
            "\n".join([point.model_dump_json() for point in retrieved_points]),
        )

        # prompt = get_prompt_for_query(query, retrieved_points)
        # write_to_file(output_folder, "prompt.xml", prompt)

        prompt = get_prompt_for_query_v2(query, retrieved_points)
        write_to_file(output_folder, "prompt.xml", prompt)

        llm_res = ollama_generate(
            ENV.LLM_QUERY_MODEL_NAME,
            LLM_QUERY_INSTRUCTION,
            prompt,
        )
        write_to_file(output_folder, "llm_response.md", llm_res)
