import json
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict


class PDFExtractor:
    """Classe pour extraire le texte des PDF (scientifiques ou financiers)"""

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)

    # ------------------------------------------------------------------
    # Méthodes d'extraction génériques (inchangées)
    # ------------------------------------------------------------------

    def extract_with_pymupdf(self):
        """Méthode 1 : Extraction rapide avec PyMuPDF"""
        text = ""
        doc = fitz.open(self.pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()

        doc.close()
        return text

    def extract_with_pdfplumber(self):
        """Méthode 2 : Meilleure pour les tableaux (texte brut uniquement)"""
        text = ""

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                page_text = page.extract_text() or ""
                text += page_text

        return text

    def extract_tables(self):
        """Extrait uniquement les tableaux du PDF (liste brute pdfplumber)"""
        tables = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)

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

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_parts = [f"\n--- Page {page_num + 1} ---\n"]

                # Texte narratif de la page
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_parts.append(page_text)

                # Tableaux de la page → Markdown structuré
                page_tables = page.extract_tables()
                if page_tables:
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

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                if not page_tables:
                    continue

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

    def _table_to_markdown(self, table: list, page_num: int, table_idx: int) -> str:
        """Convertit un tableau pdfplumber en Markdown structuré avec balises."""
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
            # Aligner sur la largeur de l'en-tête
            while len(cells) < len(header_cells):
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append(f"[/TABLE {tag}]\n")
        return "\n".join(lines)

    def _count_tables(self, text: str) -> int:
        """Compte le nombre de balises TABLE dans le texte."""
        import re
        return len(re.findall(r'\[TABLE ', text))

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save_text(self, text: str, output_path: str):
        """Sauvegarde le texte extrait."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"   ✅ Texte sauvegardé dans : {output_path}")

    def save_tables_json(self, tables: List[Dict], output_path: str):
        """Sauvegarde les tableaux JSON extraits."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Enlever 'raw' (listes imbriquées non JSON-friendly) avant sauvegarde
        clean_tables = [{k: v for k, v in t.items() if k != "raw"} for t in tables]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_tables, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Tableaux JSON sauvegardés dans : {output_path}")