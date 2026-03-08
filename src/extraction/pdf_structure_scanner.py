"""
Scan structurel rapide d'un PDF financier.

Objectif : extraire le plan du document (TOC, titres de sections, plages de pages)
en quelques dizaines de millisecondes, SANS extraire le texte complet.
Ce plan sert ensuite au SectionRouter pour cibler l'extraction sur les
pages pertinentes à la requête de l'utilisateur.

Deux sources de structure sont combinées :
  1. Bookmarks / outline PDF  (PyMuPDF, quasi-instantané)
  2. Détection heuristique de titres page par page  (pdfplumber, rapide)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


class PDFStructureScanner:
    """
    Analyse la structure d'un PDF financier sans en extraire tout le texte.

    Produit une StructuralMap :
        {
            "title": str,           # titre du document
            "total_pages": int,
            "sections": [
                {
                    "title": str,           # titre de la section
                    "start_page": int,      # 1-indexed
                    "end_page": int,        # 1-indexed, inclus
                    "level": int,           # profondeur (1 = top, 2 = sous-section…)
                    "source": str,          # "bookmark" | "heuristic"
                }
            ]
        }
    """

    # Patterns heuristiques de titres dans les rapports financiers
    _HEADING_PATTERNS = [
        # Sections courantes FR/EN
        (1, re.compile(
            r'^(Bilan|Compte de résultat|Flux de trésorerie|État des capitaux propres|'
            r'Notes annexes|Rapport de gestion|Faits marquants|Perspectives|Résultats|'
            r"Chiffre d['']affaires|Revenus|Performance|Résumé exécutif|Executive Summary|"
            r'Risques|Gouvernance|Dividendes|Endettement|Trésorerie|'
            r'Balance Sheet|Income Statement|Cash Flow|Equity|Revenue|'
            r'Risk Factors|Corporate Governance|Dividends)$',
            re.IGNORECASE
        )),
        # Titres numérotés  "1. Titre" ou "1.2 Titre"
        (2, re.compile(
            r'^\d+\.?\d*\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ\s]{3,50}$'
        )),
        # Tout-majuscule, 4+ caractères (titres de tableau ou de chapitre)
        (1, re.compile(
            r'^[A-ZÀ-Ÿ\s]{4,60}$'
        )),
    ]

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def scan(self) -> Dict:
        """
        Effectue le scan structurel complet.

        Returns:
            StructuralMap dict (voir description de classe).
        """
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)

        # 1. Titre du document (métadonnées PDF)
        doc_title = doc.metadata.get("title", "") or self.pdf_path.stem

        # 2. Bookmarks (outline) — quasi-instantané
        bookmark_sections = self._extract_bookmarks(doc, total_pages)

        doc.close()

        # 3. Si pas de bookmarks, scan heuristique des titres
        if bookmark_sections:
            sections = bookmark_sections
            print(f"   🗂️  Structure extraite via bookmarks PDF ({len(sections)} sections)")
        else:
            sections = self._heuristic_scan(total_pages)
            print(f"   🗂️  Structure extraite par analyse heuristique ({len(sections)} sections)")

        structural_map = {
            "title": doc_title,
            "total_pages": total_pages,
            "sections": sections,
        }

        return structural_map

    # ------------------------------------------------------------------
    # Source 1 : Bookmarks / Outline PDF
    # ------------------------------------------------------------------

    def _extract_bookmarks(self, doc: fitz.Document, total_pages: int) -> List[Dict]:
        """Extrait le plan (outline) intégré au PDF via PyMuPDF."""
        toc = doc.get_toc()  # [[level, title, page], ...]
        if not toc:
            return []

        sections = []
        for i, (level, title, page) in enumerate(toc):
            start = max(1, page)
            # Fin = début de la section suivante de niveau ≤ actuel, ou fin du doc
            end = total_pages
            for j in range(i + 1, len(toc)):
                next_level, _, next_page = toc[j]
                if next_level <= level:
                    end = max(start, next_page - 1)
                    break

            sections.append({
                "title": title.strip(),
                "start_page": start,
                "end_page": end,
                "level": level,
                "source": "bookmark",
            })

        return sections

    # ------------------------------------------------------------------
    # Source 2 : Analyse heuristique page par page
    # ------------------------------------------------------------------

    def _heuristic_scan(self, total_pages: int) -> List[Dict]:
        """
        Lit les premières lignes de chaque page pour détecter les titres.
        Beaucoup plus rapide qu'une extraction texte complète car on
        n'analyse que 3-5 lignes par page.
        """
        raw_headings = []  # [(page, title, level)]

        with fitz.open(str(self.pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                # Extraire seulement les premières lignes (top 15% de la page)
                r = page.rect
                clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * 0.15)
                snippet = page.get_text("text", clip=clip).strip()
                if not snippet:
                    continue

                for line in snippet.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    level = self._classify_heading(line)
                    if level is not None:
                        raw_headings.append((page_num + 1, line, level))
                        break  # un titre par page suffit

        return self._headings_to_sections(raw_headings, total_pages)

    def _classify_heading(self, line: str) -> Optional[int]:
        """Retourne le niveau du titre (1 ou 2) ou None si pas un titre."""
        for level, pattern in self._HEADING_PATTERNS:
            if pattern.match(line):
                return level
        return None

    def _headings_to_sections(
        self, headings: List[Tuple[int, str, int]], total_pages: int
    ) -> List[Dict]:
        """Convertit une liste de (page, titre, level) en sections avec plages."""
        if not headings:
            return []

        sections = []
        for i, (page, title, level) in enumerate(headings):
            end = total_pages
            for j in range(i + 1, len(headings)):
                next_page, _, next_level = headings[j]
                if next_level <= level:
                    end = max(page, next_page - 1)
                    break

            sections.append({
                "title": title,
                "start_page": page,
                "end_page": end,
                "level": level,
                "source": "heuristic",
            })

        return sections

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def get_page_range_for_section(
        self, structural_map: Dict, section_title: str
    ) -> Optional[Tuple[int, int]]:
        """
        Retourne (start_page, end_page) pour une section donnée.
        Recherche approximative (case-insensitive, substring).
        """
        query = section_title.lower()
        for section in structural_map["sections"]:
            if query in section["title"].lower():
                return section["start_page"], section["end_page"]
        return None

    def print_structural_map(self, structural_map: Dict):
        """Affiche le plan du document de façon lisible."""
        print(f"\n   📑 Plan du document : «{structural_map['title']}»")
        print(f"      {structural_map['total_pages']} pages — {len(structural_map['sections'])} sections\n")
        for s in structural_map["sections"]:
            indent = "   " * (s["level"] - 1)
            print(f"      {indent}{'└─' if s['level'] > 1 else '•'} p.{s['start_page']}-{s['end_page']}  {s['title']}")
