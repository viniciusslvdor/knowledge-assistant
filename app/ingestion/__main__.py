"""Command-line ingestion: `uv run python -m app.ingestion [--dir PATH] [--prune]`."""

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import RAGError
from app.core.logging import configure_logging
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.vector_store import ChromaVectorStore


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index documents into the knowledge base.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=settings.documents_dir,
        help=f"Folder with .txt/.md/.pdf files (default: {settings.documents_dir})",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove indexed documents that no longer exist in the folder.",
    )
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)

    try:
        pipeline = IngestionPipeline(
            SentenceTransformerEmbedder(
                settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                offline=settings.offline_mode,
            ),
            ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_file_bytes=int(settings.max_file_size_mb * 1024 * 1024),
            embedding_model_name=settings.embedding_model,
        )
        report = pipeline.ingest_directory(args.dir, prune=args.prune)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RAGError as exc:
        print(f"error: {exc.public_message} ({exc.detail})", file=sys.stderr)
        return 1

    for name, reason in report.skipped:
        print(f"skipped  {name}: {reason}")
    for name, reason in report.failed:
        print(f"failed   {name}: {reason}")
    print(report.summary())
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
