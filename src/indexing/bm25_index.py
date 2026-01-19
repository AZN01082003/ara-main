"""
Gestion de l'index BM25 pour la recherche par mots-clés
"""

import pickle
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np


class BM25Index:
    """
    Gère l'index BM25 pour la recherche par mots-clés
    """
    
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize un texte (découpe en mots)
        Simple version : split par espaces et minuscules
        """
        return text.lower().split()
    
    def index_documents(self, chunks: List[Dict]):
        """
        Indexe les documents pour BM25
        
        Args:
            chunks: Liste de chunks (avec ou sans embeddings)
        """
        print(f"   🔄 Indexation de {len(chunks)} documents dans BM25...")
        
        self.chunks = chunks
        
        # Tokeniser tous les documents
        self.tokenized_corpus = [
            self._tokenize(chunk['text']) for chunk in chunks
        ]
        
        # Créer l'index BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"   ✅ {len(chunks)} documents indexés dans BM25")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Recherche les documents les plus pertinents
        
        Args:
            query: Requête textuelle
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de résultats avec documents et scores
        """
        if self.bm25 is None:
            raise ValueError("Index BM25 non initialisé. Appelez index_documents() d'abord.")
        
        # Tokeniser la requête
        tokenized_query = self._tokenize(query)
        
        # Calculer les scores BM25
        scores = self.bm25.get_scores(tokenized_query)
        
        # Obtenir les top_k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Formater les résultats
        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': self.chunks[idx]['chunk_id'],
                'text': self.chunks[idx]['text'],
                'metadata': self.chunks[idx].get('metadata', {}),
                'bm25_score': float(scores[idx])
            })
        
        return results
    
    def save_index(self, output_path: str):
        """Sauvegarde l'index BM25"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"   ✓ Index BM25 sauvegardé : {output_path}")
    
    def load_index(self, input_path: str):
        """Charge l'index BM25"""
        with open(input_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.chunks = index_data['chunks']
        self.tokenized_corpus = index_data['tokenized_corpus']
        
        print(f"   ✓ Index BM25 chargé : {input_path}")
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur l'index"""
        if self.bm25 is None:
            return {'indexed': False}
        
        return {
            'indexed': True,
            'total_documents': len(self.chunks),
            'avg_doc_length': np.mean([len(doc) for doc in self.tokenized_corpus])
        }