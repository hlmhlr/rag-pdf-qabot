"""Tests for loading, checking and splitting PDFs."""

import pytest
from conftest import HANDBOOK, MANUAL
from pypdf import PdfWriter

from qabot import ingest
from qabot.ingest import IngestError, check_files, load_pdfs, split_documents


def test_loads_every_page_of_several_pdfs():
    pages = load_pdfs([HANDBOOK, MANUAL])
    assert len(pages) == 6  # 3 pages each
    assert {p.metadata["file"] for p in pages} == {HANDBOOK.name, MANUAL.name}
    assert [p.metadata["page"] for p in pages[:3]] == [0, 1, 2]


def test_chunks_respect_size_and_keep_metadata():
    chunks = split_documents(load_pdfs([HANDBOOK]), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3  # 3 pages became more than 3 chunks
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.metadata["file"] == HANDBOOK.name and "page" in c.metadata for c in chunks)


def test_rejects_empty_list_and_non_pdf(tmp_path):
    with pytest.raises(IngestError, match="at least one"):
        check_files([])
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    with pytest.raises(IngestError, match="not a PDF"):
        check_files([notes])


def test_rejects_too_many_files():
    with pytest.raises(IngestError, match="at most"):
        check_files([HANDBOOK] * (ingest.MAX_FILES + 1))


def test_rejects_too_large_upload(monkeypatch):
    monkeypatch.setattr(ingest, "MAX_TOTAL_MB", 0.001)  # 1 KB limit, the samples are ~4 KB
    with pytest.raises(IngestError, match="limit"):
        check_files([HANDBOOK])


def test_rejects_pdf_without_text(tmp_path):
    scanned = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)  # a page with no text, like a scan
    writer.write(scanned)
    with pytest.raises(IngestError, match="no extractable text"):
        load_pdfs([scanned])


def test_rejects_broken_pdf(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not really a pdf")
    with pytest.raises(IngestError, match="could not be read"):
        load_pdfs([broken])
