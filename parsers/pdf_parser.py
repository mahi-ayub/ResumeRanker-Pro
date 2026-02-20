"""PDF Resume Parser — extracts raw text from PDF files.

Uses pdfplumber as primary extractor with PyMuPDF (fitz) as fallback.
Handles multi-page resumes, scanned PDFs, and malformed files gracefully.
"""

from typing import Optional
from pathlib import Path
from loguru import logger


class PDFParser:
    """Extract text from PDF resume files."""

    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages

    def extract_text(self, file_path: Optional[str] = None, file_bytes: Optional[bytes] = None) -> str:
        """
        Extract text from a PDF file path or raw bytes.

        Args:
            file_path: Path to the PDF file.
            file_bytes: Raw PDF bytes (e.g., from Streamlit upload).

        Returns:
            Extracted text string.

        Raises:
            ValueError: If neither file_path nor file_bytes is provided.
        """
        if file_path is None and file_bytes is None:
            raise ValueError("Provide either file_path or file_bytes.")

        # Try pdfplumber first (better for text-heavy PDFs)
        text = self._extract_with_pdfplumber(file_path, file_bytes)

        # Fallback to PyMuPDF if pdfplumber yields too little text
        if len(text.strip()) < 50:
            logger.info("pdfplumber extraction insufficient, falling back to PyMuPDF.")
            text_fitz = self._extract_with_pymupdf(file_path, file_bytes)
            if len(text_fitz.strip()) > len(text.strip()):
                text = text_fitz

        # If both methods give short text, try combining them
        if len(text.strip()) < 100:
            text_fitz = self._extract_with_pymupdf(file_path, file_bytes)
            if len(text_fitz.strip()) > len(text.strip()):
                text = text_fitz

        if len(text.strip()) < 20:
            logger.warning("Extracted very little text from PDF. The file may be image-based.")

        # Clean up common PDF extraction artifacts
        text = self._clean_pdf_text(text)

        return text

    def _clean_pdf_text(self, text: str) -> str:
        """Clean up common PDF extraction artifacts."""
        import re
        # Fix multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Fix broken lines within sentences (line break in middle of sentence)
        # But preserve intentional line breaks (after periods, bullets, etc.)
        text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)
        # Normalize unicode dashes to regular dashes
        text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
        # Remove null bytes
        text = text.replace('\x00', '')
        return text

    def _extract_with_pdfplumber(
        self, file_path: Optional[str] = None, file_bytes: Optional[bytes] = None
    ) -> str:
        """Extract text using pdfplumber."""
        try:
            import pdfplumber
            import io

            source = file_path if file_path else io.BytesIO(file_bytes)
            pages_text = []

            with pdfplumber.open(source) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= self.max_pages:
                        break
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)

            return "\n\n".join(pages_text)

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""

    def _extract_with_pymupdf(
        self, file_path: Optional[str] = None, file_bytes: Optional[bytes] = None
    ) -> str:
        """Extract text using PyMuPDF (fitz) as fallback."""
        try:
            import fitz  # PyMuPDF

            if file_path:
                doc = fitz.open(file_path)
            else:
                doc = fitz.open(stream=file_bytes, filetype="pdf")

            pages_text = []
            for i, page in enumerate(doc):
                if i >= self.max_pages:
                    break
                pages_text.append(page.get_text())

            doc.close()
            return "\n\n".join(pages_text)

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""

    def extract_from_text_file(self, file_path: Optional[str] = None, file_bytes: Optional[bytes] = None) -> str:
        """Extract text from a plain text file."""
        if file_path:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        elif file_bytes:
            return file_bytes.decode("utf-8", errors="ignore")
        raise ValueError("Provide either file_path or file_bytes.")
