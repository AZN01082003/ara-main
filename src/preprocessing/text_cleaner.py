"""
Nettoyage du texte extrait des PDF
"""

import re
from unidecode import unidecode


class TextCleaner:
    """Nettoie et normalise le texte extrait (mode générique ou financier)"""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Pipeline générique (articles scientifiques)
    # ------------------------------------------------------------------

    def clean(self, text):
        """Pipeline de nettoyage complet (mode standard)"""

        # Étape 1 : Supprimer les caractères spéciaux non-textuels
        text = self._remove_special_chars(text)

        # Étape 2 : Normaliser les espaces
        text = self._normalize_spaces(text)

        # Étape 3 : Fusionner les mots coupés en fin de ligne
        text = self._fix_hyphenation(text)

        # Étape 4 : Nettoyer les en-têtes/pieds de page
        text = self._remove_headers_footers(text)

        # Étape 5 : Normaliser la ponctuation
        text = self._normalize_punctuation(text)

        return text

    def _remove_special_chars(self, text):
        """Enlève les caractères non-textuels (mode standard)."""
        # Garde lettres, chiffres, ponctuation de base
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\'\"\n]', ' ', text)
        return text

    def _normalize_spaces(self, text):
        """Normalise les espaces multiples"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()

    def _fix_hyphenation(self, text):
        """
        Fusionne les mots coupés en fin de ligne
        Exemple: "connec-\ntion" → "connection"
        """
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text

    def _remove_headers_footers(self, text):
        """Supprime les en-têtes/pieds de page répétitifs."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Ignorer les lignes qui sont juste des numéros
            if re.match(r'^\s*\d+\s*$', line):
                continue
            # Ignorer les lignes très courtes (probablement des en-têtes)
            if len(line.strip()) < 3:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _normalize_punctuation(self, text):
        """Normalise la ponctuation"""
        text = re.sub(r'([\.,:;!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
        return text

    # ------------------------------------------------------------------
    # Pipeline financier — préserve les symboles monétaires et tableaux
    # ------------------------------------------------------------------

    def clean_financial(self, text: str) -> str:
        """
        Pipeline de nettoyage adapté aux rapports financiers.

        Différences vs clean() :
        - Préserve les symboles : %, €, $, |, +, /, M€, k€, …
        - Ne touche PAS aux blocs tableau Markdown ([TABLE]…[/TABLE])
        - Supprime uniquement les vrais caractères de contrôle ASCII
        - Évite de fusionner des mots à travers les séparateurs de tableau
        """
        # Étape 1 : Caractères de contrôle seulement (pas les symboles financiers)
        text = self._remove_control_chars(text)

        # Étape 2 : Normaliser les espaces en respectant les tableaux
        text = self._normalize_spaces_financial(text)

        # Étape 3 : Fusionner les mots coupés (hors lignes de tableau)
        text = self._fix_hyphenation_financial(text)

        # Étape 4 : En-têtes/pieds de page (idem standard)
        text = self._remove_headers_footers_financial(text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """
        Supprime uniquement les caractères de contrôle ASCII (0x00-0x08,
        0x0B-0x0C, 0x0E-0x1F, 0x7F).  Préserve %, €, $, |, +, /, etc.
        """
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def _normalize_spaces_financial(self, text: str) -> str:
        """Normalise les espaces en ne touchant pas aux lignes de tableau."""
        lines = text.split('\n')
        normalized = []

        for line in lines:
            stripped = line.strip()
            # Lignes de tableau Markdown et balises → inchangées
            if stripped.startswith('|') or stripped.startswith('[TABLE') or stripped.startswith('[/TABLE'):
                normalized.append(line)
            else:
                normalized.append(re.sub(r' +', ' ', line))

        text = '\n'.join(normalized)
        # Réduire les sauts de ligne excessifs (hors blocs tableau)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _fix_hyphenation_financial(self, text: str) -> str:
        """
        Fusionne les mots coupés en fin de ligne, sans toucher aux lignes
        de tableau Markdown ni aux balises [TABLE].
        """
        lines = text.split('\n')
        result = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Lignes de tableau → ne pas fusionner
            if stripped.startswith('|') or stripped.startswith('[TABLE') or stripped.startswith('[/TABLE'):
                result.append(line)
                i += 1
                continue

            # Détecter un mot coupé en fin de ligne (lettre-tiret en fin)
            if (
                i + 1 < len(lines)
                and re.search(r'\w-$', line.rstrip())
                and lines[i + 1].strip()
                and not lines[i + 1].strip().startswith('|')
            ):
                fused = re.sub(r'(\w)-$', r'\1', line.rstrip()) + lines[i + 1].lstrip()
                result.append(fused)
                i += 2
                continue

            result.append(line)
            i += 1

        return '\n'.join(result)

    def _remove_headers_footers_financial(self, text: str) -> str:
        """
        Supprime les numéros de page isolés et lignes trop courtes,
        en préservant les lignes de tableau Markdown.
        """
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Préserver les lignes de tableau et balises
            if stripped.startswith('|') or stripped.startswith('[TABLE') or stripped.startswith('[/TABLE'):
                cleaned_lines.append(line)
                continue

            # Ignorer numéros de page isolés
            if re.match(r'^\s*\d+\s*$', line):
                continue

            # Ignorer lignes très courtes (hors tableau)
            if len(stripped) < 3:
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    # ------------------------------------------------------------------
    # Utilitaires communs
    # ------------------------------------------------------------------

    def clean_for_search(self, text: str) -> str:
        """Nettoyage pour la recherche BM25 (minuscules + sans accents)"""
        text = text.lower()
        text = unidecode(text)
        return text