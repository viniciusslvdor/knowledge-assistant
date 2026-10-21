"""Read TXT, Markdown and PDF files into raw text pages."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


class DocumentLoadError(Exception):
    """The file exists but its text could not be extracted."""


@dataclass(frozen=True)
class Page:
    text: str
    page: int | None  # 1-based page number for PDFs, None for plain text


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    filename: str  # path relative to the documents directory, with "/" separators
    pages: list[Page]


def make_document_id(relative_name: str) -> str:
    """Stable ID derived from the file's relative path, so re-ingesting the
    same file updates it instead of creating a duplicate."""
    digest = hashlib.sha256(relative_name.lower().encode("utf-8")).hexdigest()
    return f"doc-{digest[:16]}"


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise DocumentLoadError("file looks binary, not text")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("unsupported text encoding")


def _read_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError("PDF is password protected")
        pages = []
        for number, pdf_page in enumerate(reader.pages, start=1):
            pages.append(Page(text=pdf_page.extract_text() or "", page=number))
        return pages
    except DocumentLoadError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as exc:
        raise DocumentLoadError(f"unreadable PDF ({type(exc).__name__})") from exc


def load_document(path: Path, base_dir: Path) -> LoadedDocument:
    relative_name = path.resolve().relative_to(base_dir.resolve()).as_posix()
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_EXTENSIONS:
            pages = [Page(text=_read_text_file(path), page=None)]
        elif suffix in PDF_EXTENSIONS:
            pages = _read_pdf(path)
        else:
            raise DocumentLoadError(f"unsupported file type '{suffix}'")
    except OSError as exc:
        raise DocumentLoadError(f"cannot read file ({type(exc).__name__})") from exc
    return LoadedDocument(
        document_id=make_document_id(relative_name), filename=relative_name, pages=pages
    )
