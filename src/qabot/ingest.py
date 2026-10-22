"""Load PDF files and split them into chunks (lab: document_loader, text_splitter).

The lab loads one PDF per question. Here several PDFs are loaded together,
each page is labelled with its file name, and the input is checked first,
so a bad upload gets a clear message instead of a crash.

Why not PyPDFLoader
    The lab imports PyPDFLoader from langchain-community, which LangChain is
    sunsetting (no longer maintained), and the loader has no standalone
    package yet. PyPDFLoader is a thin wrapper around pypdf, so read_pdf()
    calls pypdf directly and returns the same thing: one Document per page,
    with "source" and "page" (counted from 0) metadata.

Contents
    check_files()      PDF-only, at most MAX_FILES files and MAX_TOTAL_MB in total
    read_pdf()         one Document per page of one PDF (replaces PyPDFLoader)
    load_pdfs()        all pages of several PDFs, each labelled with its file name
    split_documents()  the lab's RecursiveCharacterTextSplitter (1000 / 50)
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Upload limits. Large documents "will fail with the current setup" in the
# lab; here they are refused up front with a message. 20 MB keeps embedding
# time reasonable on the free CPU deployment.
MAX_FILES = 5
MAX_TOTAL_MB = 20

# The lab's splitter settings.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50


class IngestError(ValueError):
    """An upload problem the user can fix (wrong type, too large, no text)."""


def check_files(paths: list[str | Path]) -> list[Path]:
    """Validate the uploaded files and return them as Path objects."""
    files = [Path(p) for p in paths]
    if not files:
        raise IngestError("Upload at least one PDF file.")
    if len(files) > MAX_FILES:
        raise IngestError(f"Upload at most {MAX_FILES} PDF files at a time ({len(files)} given).")

    for file in files:
        if file.suffix.lower() != ".pdf":
            raise IngestError(f"{file.name} is not a PDF file.")

    total_mb = sum(file.stat().st_size for file in files) / 1_000_000
    if total_mb > MAX_TOTAL_MB:
        raise IngestError(f"The files total {total_mb:.1f} MB; the limit is {MAX_TOTAL_MB} MB.")
    return files


def read_pdf(path: Path) -> list[Document]:
    """Read one PDF into one Document per page (what PyPDFLoader.load() returns)."""
    reader = PdfReader(path)
    return [
        Document(
            page_content=(page.extract_text() or "").strip(),  # PyPDFLoader strips too
            metadata={"source": str(path), "page": number},
        )
        for number, page in enumerate(reader.pages)
    ]


def load_pdfs(paths: list[str | Path]) -> list[Document]:
    """Load every page of every PDF as a Document.

    Each page also gets "file" (the file name without the temporary upload
    folder), used later to show where an answer came from.
    """
    pages = []
    for file in check_files(paths):
        try:
            file_pages = read_pdf(file)
        except Exception as error:  # pypdf raises several types for broken files
            raise IngestError(f"{file.name} could not be read as a PDF ({error}).") from error

        if not any(page.page_content.strip() for page in file_pages):
            raise IngestError(
                f"{file.name} has no extractable text. It may be a scanned image, "
                "which would need OCR first."
            )

        for page in file_pages:
            page.metadata["file"] = file.name
        pages.extend(file_pages)
    return pages


def split_documents(
    pages: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split pages into overlapping chunks; each chunk keeps its page's metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(pages)
