import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import unicodedata

import fitz


def extract_text_without_tables_and_figures(page: fitz.Page) -> str:
    page_dict = page.get_text("dict")

    # Rectangles des tableaux
    table_rects = []
    try:
        tables = page.find_tables()
        #table_rects = [fitz.Rect(t.bbox) for t in tables.tables]
        table_rects = []

        for t in tables.tables:
          table_rects.append(fitz.Rect(t.bbox))

    except Exception:
        pass

    clean_text = []

    for block in page_dict["blocks"]:

        # uniquement les blocs texte
        if block["type"] != 0:
            continue

        block_rect = fitz.Rect(block["bbox"])

        # supprimer les blocs appartenant à un tableau
        if any(block_rect.intersects(r) for r in table_rects):
            continue

        text = ""

        for line in block["lines"]:
            for span in line["spans"]:
                text += span["text"]

        text = text.strip()

        # supprimer les légendes
        if re.match(r"^(Figure|Fig\.?|Table)\s*\d+", text, re.IGNORECASE):
            continue

        clean_text.append(text)

    return "\n".join(clean_text)



def clean_pdf_text(text: str) -> str:
    # 0. Normaliser les caractères unicode (très important pour les points PDF)
    text = unicodedata.normalize("NFKC", text)

    # NEW STEP 1: Convert space-separated decimals to dot-separated for units (million, %) (e.g., "9 3%" -> "9.3%")
    # The lookahead ensures we only modify the space *between* the numbers if followed by a unit.
    text = re.sub(r'(\d+)\s(\d+)(?=\s*(?:million|%|k|M|B)(?=\W|$))', r'\1.\2', text, flags=re.IGNORECASE)

    # NEW STEP 2: Convert space-separated decimals to dot-separated for currency amounts (e.g., "€ 48 73" -> "€ 48.73")
    # Captures currency symbol and optional space, then two number parts.
    text = re.sub(r'((?:€|\$)\s*)(\d+)\s(\d+)', r'\1\2.\3', text)

    # 1. Supprimer tous les points et assimilés (Unicode category: Po = punctuation other) BUT preserve decimal points
    # Modified to NOT remove the period '.' character if it is a decimal point.
    text = re.sub(r'[\u2022\u2027\u2219\u00B7\u2043\u25CF\u25E6\u2024\uFE52\uFF0E]+', ' ', text)

    # 2. Supprimer les traits d’union PDF
    text = re.sub(r"-\s+", "", text)

    # 3. Supprimer les multiples espaces
    text = re.sub(r"\s+", " ", text)

    # 4. Supprimer les séparateurs horizontaux
    text = re.sub(r"[_=]{3,}", " ", text)

    # 5. Supprimer les numéros de page
    text = re.sub(r"Page\s*\d+\s*/\s*\d+", " ", text, flags=re.IGNORECASE)

    # 6. Nettoyage final
    text = text.strip()
    return text


import fitz
from langchain_core.documents import Document

class PdfIngestion:

    def __init__(self, chunk_size = 1000, chunk_overlap = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def extract_only_text(self,page:fitz.Page) -> str:
      return extract_text_without_tables_and_figures(page)



    def _clean(self, text : str) -> str:
        return clean_pdf_text(text)

    def process(self, path : str):

      raw_docs = []
      with fitz.open(path) as pdf_doc:
        for i, page in enumerate(pdf_doc):
              extracted_text = self.extract_only_text(page)
            # Create a LangChain Document object for each page
              doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": path,
                    "page": i
                }
            )
              raw_docs.append(doc)


        # Now clean the page content of the Documents
      for d in raw_docs:
            d.page_content = self._clean(d.page_content)

      chunks = self.splitter.split_documents(raw_docs)

      return chunks