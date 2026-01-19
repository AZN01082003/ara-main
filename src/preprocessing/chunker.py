"""
Découpage sémantique du texte en chunks
"""

import json
import re
from pathlib import Path
from typing import List, Dict
#from langchain.text_splitter import RecursiveCharacterTextSplitter
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