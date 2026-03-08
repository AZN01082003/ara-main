"""
PDFExtractor — extraction texte + tableaux via PyMuPDF (fitz) uniquement.

Remplace l'ancienne dépendance pdfplumber/pdfminer.six (qui amenait cryptography,
incompatible avec Python 3.10.0rc1 via PyO3 abi3).
PyMuPDF >= 1.23 offre page.find_tables() avec un format de sortie identique
à pdfplumber.extract_tables() : list[list[list[str | None]]].
"""

import json
import re
import fitz  # PyMuPDF — déjà requis pour le scan structurel
from pathlib import Path
from typing import List, Dict


class PDFExtractor:
    """Extraction texte + tableaux depuis un PDF (scientifique ou financier)."""

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)

    # ------------------------------------------------------------------
    # Méthodes d'extraction génériques
    # ------------------------------------------------------------------

    def extract_with_pymupdf(self) -> str:
        """Extraction rapide du texte brut page par page."""
        text = ""
        with fitz.open(str(self.pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text("text")
        return text

    def extract_with_pdfplumber(self) -> str:
        """Alias de extract_with_pymupdf (compatibilité API publique)."""
        return self.extract_with_pymupdf()

    def extract_tables(self) -> list:
        """Extrait tous les tableaux du PDF (format brut : list[list[list]])."""
        tables = []
        with fitz.open(str(self.pdf_path)) as doc:
            for page in doc:
                tables.extend(self._get_page_tables(page))
        return tables

    # ------------------------------------------------------------------
    # Méthodes spécialisées pour les rapports financiers
    # ------------------------------------------------------------------

    def extract_financial_report(self) -> str:
        """
        Extraction enrichie pour les rapports financiers.

        Combine texte narratif et tableaux sérialisés en Markdown,
        insérés à leur position exacte dans la page.  Les tableaux sont
        délimités par des balises [TABLE p.X #Y] … [/TABLE p.X #Y]
        pour permettre un chunking atomique en aval.

        Returns:
            Texte complet avec tableaux Markdown intégrés.
        """
        pages_content = []

        with fitz.open(str(self.pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                page_parts = [f"\n--- Page {page_num + 1} ---\n"]

                # Texte narratif de la page
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    page_parts.append(page_text)

                # Tableaux de la page → Markdown structuré
                page_tables = self._get_page_tables(page)
                for t_idx, table in enumerate(page_tables):
                    md = self._table_to_markdown(table, page_num + 1, t_idx + 1)
                    if md:
                        page_parts.append(md)

                pages_content.append("\n".join(page_parts))

        full_text = "\n".join(pages_content)
        print(f"   📊 {self._count_tables(full_text)} tableaux intégrés en Markdown")
        return full_text

    def extract_tables_as_json(self) -> List[Dict]:
        """
        Extrait tous les tableaux du PDF en format JSON structuré.

        Returns:
            Liste de dicts avec clés : table_id, page, headers, rows, raw
        """
        all_tables = []

        with fitz.open(str(self.pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                page_tables = self._get_page_tables(page)

                for t_idx, table in enumerate(page_tables):
                    if not table or len(table) == 0:
                        continue

                    # Première ligne = en-têtes
                    headers = [
                        str(cell).strip() if cell else f"col_{i}"
                        for i, cell in enumerate(table[0])
                    ]

                    rows = []
                    for row in table[1:]:
                        row_dict = {}
                        for j, cell in enumerate(row):
                            col_name = headers[j] if j < len(headers) else f"col_{j}"
                            row_dict[col_name] = str(cell).strip() if cell else ""
                        rows.append(row_dict)

                    all_tables.append({
                        "table_id": f"table_p{page_num + 1}_{t_idx + 1}",
                        "page": page_num + 1,
                        "table_index": t_idx + 1,
                        "headers": headers,
                        "rows": rows,
                        "raw": table,
                    })

        return all_tables

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_page_tables(self, page) -> list:
        """
        Extrait les tableaux d'une page fitz.
        Retourne list[list[list[str|None]]] — même format que pdfplumber.
        """
        try:
            finder = page.find_tables()
            return [t.extract() for t in finder.tables]
        except Exception:
            return []

    def _table_to_markdown(self, table: list, page_num: int, table_idx: int) -> str:
        """Convertit un tableau en Markdown structuré avec balises."""
        if not table or len(table) == 0:
            return ""

        tag = f"p.{page_num} #{table_idx}"
        lines = [f"\n[TABLE {tag}]"]

        # En-tête (première ligne)
        header_cells = [str(cell).strip() if cell else "" for cell in table[0]]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

        # Lignes de données
        for row in table[1:]:
            cells = [str(cell).strip() if cell else "" for cell in row]
            while len(cells) < len(header_cells):
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append(f"[/TABLE {tag}]\n")
        return "\n".join(lines)

    def _count_tables(self, text: str) -> int:
        return len(re.findall(r'\[TABLE ', text))

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save_text(self, text: str, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"   ✅ Texte sauvegardé dans : {output_path}")

    def save_tables_json(self, tables: List[Dict], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_tables = [{k: v for k, v in t.items() if k != "raw"} for t in tables]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_tables, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Tableaux JSON sauvegardés dans : {output_path}")
