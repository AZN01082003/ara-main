"""
Découpage sémantique du texte en chunks
(mode générique + mode financier)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

import spacy


class SemanticChunker:
    """
    Découpe le texte en chunks sémantiques
    (paragraphes, sections logiques)
    """
    
    def __init__(self, chunk_size=500, overlap=50, language='en'):
        """
        Args:
            chunk_size: Taille cible d'un chunk (en caractères)
            overlap: Chevauchement entre chunks (pour garder le contexte)
            language: 'en' pour anglais, 'fr' pour français
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.language = language
        
        # Charger le modèle spaCy
        if language == 'en':
            self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = spacy.load('fr_core_news_sm')
        
        # Splitter de LangChain
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Découpe le texte en chunks intelligents
        
        Returns:
            Liste de dictionnaires avec:
            - text: le contenu du chunk
            - chunk_id: identifiant unique
            - metadata: infos supplémentaires
        """
        
        # Méthode 1 : Découpage par sections (si détectées)
        sections = self._detect_sections(text)
        
        if sections:
            print(f"   📑 {len(sections)} sections détectées")
            chunks = self._chunk_by_sections(sections)
        else:
            print(f"   📄 Aucune section claire, découpage récursif")
            chunks = self._chunk_recursive(text)
        
        # Ajouter métadonnées
        chunks_with_metadata = []
        for i, chunk_text in enumerate(chunks):
            chunks_with_metadata.append({
                'chunk_id': f'chunk_{i:04d}',
                'text': chunk_text,
                'length': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'metadata': {
                    'position': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return chunks_with_metadata
    
    def _detect_sections(self, text: str) -> List[Dict]:
        """
        Détecte les sections dans un article scientifique
        (Abstract, Introduction, Methods, Results, etc.)
        """
        sections = []
        
        # Patterns de titres de sections courants
        section_patterns = [
            r'\n\s*(Abstract|ABSTRACT)\s*\n',
            r'\n\s*(Introduction|INTRODUCTION)\s*\n',
            r'\n\s*(Methods?|METHODS?|Methodology|METHODOLOGY)\s*\n',
            r'\n\s*(Results?|RESULTS?)\s*\n',
            r'\n\s*(Discussion|DISCUSSION)\s*\n',
            r'\n\s*(Conclusion|CONCLUSION)\s*\n',
            r'\n\s*(References|REFERENCES)\s*\n',
            r'\n\s*(\d+\.?\s+[A-Z][a-z]+.*)\n',  # Sections numérotées
        ]
        
        # Combiner tous les patterns
        combined_pattern = '|'.join(section_patterns)
        
        # Trouver toutes les correspondances
        matches = list(re.finditer(combined_pattern, text))
        
        if not matches:
            return []
        
        # Extraire les sections
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_title = match.group().strip()
            section_content = text[start:end].strip()
            
            sections.append({
                'title': section_title,
                'content': section_content,
                'start': start,
                'end': end
            })
        
        return sections
    
    def _chunk_by_sections(self, sections: List[Dict]) -> List[str]:
        """Découpe chaque section individuellement"""
        all_chunks = []
        
        for section in sections:
            # Si la section est petite, la garder entière
            if len(section['content']) <= self.chunk_size * 1.5:
                all_chunks.append(section['content'])
            else:
                # Sinon, la découper
                section_chunks = self.splitter.split_text(section['content'])
                all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _chunk_recursive(self, text: str) -> List[str]:
        """Découpage récursif classique"""
        return self.splitter.split_text(text)
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk=5) -> List[str]:
        """
        Méthode alternative : découper par phrases
        (plus précis mais plus lent)
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict], output_path: str):
        """Sauvegarde les chunks en JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Chunks sauvegardés : {output_path}")
    
    def load_chunks(self, input_path: str) -> List[Dict]:
        """Charge les chunks depuis un JSON"""
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks
    
    def get_statistics(self, chunks: List[Dict]) -> Dict:
        """Calcule des statistiques sur les chunks"""
        lengths = [chunk['length'] for chunk in chunks]
        word_counts = [chunk['word_count'] for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_words': sum(word_counts) / len(word_counts),
            'total_words': sum(word_counts)
        }


# ==============================================================
# Découpeur spécialisé pour les rapports financiers
# ==============================================================

class FinancialChunker(SemanticChunker):
    """
    Découpe le texte de rapports financiers en chunks intelligents.

    Améliorations vs SemanticChunker :
    - Reconnaît les sections financières (Bilan, CdR, Flux, etc.)
    - Garde les blocs tableau Markdown intacts (atomiques, non découpés)
    - Tague chaque chunk : 'table' | 'narrative'
    - Chunk size plus grand par défaut (tableaux = beaucoup de texte)
    """

    # Patterns de sections typiques des rapports financiers (FR + EN)
    FINANCIAL_SECTION_PATTERNS = [
        r'\n\s*(Bilan|BILAN)\s*\n',
        r'\n\s*(Compte de résultat|COMPTE DE RÉSULTAT|Compte de Résultat)\s*\n',
        r'\n\s*(Flux de trésorerie|FLUX DE TRÉSORERIE|Tableau des flux)\s*\n',
        r'\n\s*(État des capitaux propres|CAPITAUX PROPRES|Capitaux propres)\s*\n',
        r'\n\s*(Notes annexes|NOTES ANNEXES|Annexes|ANNEXES)\s*\n',
        r'\n\s*(Rapport de gestion|RAPPORT DE GESTION)\s*\n',
        r'\n\s*(Faits marquants|FAITS MARQUANTS)\s*\n',
        r'\n\s*(Perspectives|PERSPECTIVES|Outlook)\s*\n',
        r'\n\s*(Résultats|RÉSULTATS|Results)\s*\n',
        r'\n\s*(Chiffre d[\'']affaires|CHIFFRE D[\'']AFFAIRES|Revenus|REVENUS)\s*\n',
        r'\n\s*(Performance|PERFORMANCE)\s*\n',
        r'\n\s*(Résumé exécutif|RÉSUMÉ EXÉCUTIF|Executive Summary)\s*\n',
        r'\n\s*(Risques|RISQUES|Risk Factors)\s*\n',
        r'\n\s*(Gouvernance|GOUVERNANCE|Corporate Governance)\s*\n',
        r'\n\s*(Dividendes|DIVIDENDES)\s*\n',
        r'\n\s*(Endettement|ENDETTEMENT|Dette|DETTE)\s*\n',
        r'\n\s*(Trésorerie|TRÉSORERIE)\s*\n',
        # Sections numérotées génériques (1. Titre, 2.1 Titre…)
        r'\n\s*(\d+\.?\d*\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ\s]{3,})\s*\n',
    ]

    # Regex pour détecter les balises TABLE insérées par PDFExtractor
    _TABLE_BLOCK_RE = re.compile(
        r'\[TABLE [^\]]+\].*?\[/TABLE [^\]]+\]',
        re.DOTALL
    )

    def __init__(self, chunk_size: int = 800, overlap: int = 100, language: str = 'fr'):
        """
        Args:
            chunk_size: Taille cible d'un chunk (en caractères).
                        Plus grand que le défaut car les tableaux sont denses.
            overlap: Chevauchement entre chunks narratifs consécutifs.
            language: 'fr' (défaut pour rapports FR) ou 'en'.
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap, language=language)

        self._financial_pattern = re.compile(
            '|'.join(self.FINANCIAL_SECTION_PATTERNS)
        )

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def chunk_financial_text(self, text: str) -> List[Dict]:
        """
        Découpe un texte financier enrichi (avec balises [TABLE]…[/TABLE]).

        Stratégie :
        1. Séparer les blocs tableau des blocs texte narratif
        2. Découper le texte narratif par sections financières détectées
        3. Garder les tableaux comme chunks atomiques
        4. Réunir le tout dans l'ordre de position dans le document

        Returns:
            Liste de chunks avec métadonnées (chunk_type, section…)
        """
        text_blocks, table_blocks = self._split_tables_from_text(text)

        narrative_chunks = self._chunk_narrative_blocks(text_blocks)
        table_chunk_objects = self._create_table_chunks(table_blocks)

        all_raw = narrative_chunks + table_chunk_objects
        all_raw.sort(key=lambda x: x['position'])

        return self._finalize_chunks(all_raw)

    # ------------------------------------------------------------------
    # Séparation texte / tableaux
    # ------------------------------------------------------------------

    def _split_tables_from_text(self, text: str):
        """Sépare le texte en blocs narratifs et blocs tableau."""
        table_blocks = []
        text_blocks = []
        last_end = 0

        for match in self._TABLE_BLOCK_RE.finditer(text):
            before = text[last_end:match.start()]
            if before.strip():
                text_blocks.append({'content': before, 'position': last_end})
            table_blocks.append({'content': match.group(), 'position': match.start()})
            last_end = match.end()

        remaining = text[last_end:]
        if remaining.strip():
            text_blocks.append({'content': remaining, 'position': last_end})

        return text_blocks, table_blocks

    # ------------------------------------------------------------------
    # Chunking du narratif financier
    # ------------------------------------------------------------------

    def _chunk_narrative_blocks(self, text_blocks: list) -> list:
        """Découpe chaque bloc narratif par sections financières."""
        chunks = []

        for block in text_blocks:
            raw_text = block['content']
            sections = self._detect_financial_sections(raw_text)

            if sections:
                for section in sections:
                    self._chunk_section(section, block['position'], chunks)
            else:
                # Pas de sections détectées → découpage récursif classique
                for sub in self.splitter.split_text(raw_text):
                    chunks.append({
                        'content': sub,
                        'type': 'narrative',
                        'section': None,
                        'position': block['position'],
                    })

        return chunks

    def _detect_financial_sections(self, text: str) -> list:
        """Identifie les sections financières dans un bloc de texte."""
        matches = list(self._financial_pattern.finditer(text))
        if not matches:
            return []

        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append({
                'title': match.group().strip(),
                'content': text[start:end].strip(),
                'start': start,
            })
        return sections

    def _chunk_section(self, section: dict, block_offset: int, out: list):
        """Découpe une section (potentiellement longue) en sous-chunks."""
        content = section['content']
        pos = block_offset + section['start']

        if len(content) <= self.chunk_size * 1.5:
            out.append({
                'content': content,
                'type': 'narrative',
                'section': section['title'],
                'position': pos,
            })
        else:
            for sub in self.splitter.split_text(content):
                out.append({
                    'content': sub,
                    'type': 'narrative',
                    'section': section['title'],
                    'position': pos,
                })

    # ------------------------------------------------------------------
    # Chunks atomiques pour les tableaux
    # ------------------------------------------------------------------

    def _create_table_chunks(self, table_blocks: list) -> list:
        """Crée un chunk atomique (non découpé) par bloc tableau."""
        return [
            {
                'content': block['content'],
                'type': 'table',
                'section': None,
                'position': block['position'],
            }
            for block in table_blocks
        ]

    # ------------------------------------------------------------------
    # Finalisation et métadonnées
    # ------------------------------------------------------------------

    def _finalize_chunks(self, raw_chunks: list) -> List[Dict]:
        """Attribue les IDs et métadonnées finales à chaque chunk."""
        result = []
        for i, chunk in enumerate(raw_chunks):
            result.append({
                'chunk_id': f'chunk_{i:04d}',
                'text': chunk['content'],
                'length': len(chunk['content']),
                'word_count': len(chunk['content'].split()),
                'metadata': {
                    'position': i,
                    'total_chunks': len(raw_chunks),
                    'chunk_type': chunk.get('type', 'narrative'),
                    'section': chunk.get('section'),
                    'is_table': chunk.get('type') == 'table',
                },
            })
        return result

    # ------------------------------------------------------------------
    # Surcharge pour compatibilité avec le pipeline générique
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Surcharge de SemanticChunker.chunk_text().
        Appelle chunk_financial_text() pour bénéficier du traitement
        spécialisé même quand le code appelant n'est pas modifié.
        """
        return self.chunk_financial_text(text)