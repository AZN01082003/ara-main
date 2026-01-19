"""
Système de recherche hybride combinant Chroma et BM25
"""

import numpy as np
from typing import List, Dict, Optional
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.embeddings.embedding_generator import EmbeddingGenerator


class HybridRetriever:
    """
    Combine la recherche vectorielle (Chroma) et la recherche par mots-clés (BM25)
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_generator: EmbeddingGenerator,
        alpha: float = 0.5,
        top_k: int = 5
    ):
        """
        Args:
            vector_store: Instance de VectorStore (Chroma)
            bm25_index: Instance de BM25Index
            embedding_generator: Générateur d'embeddings pour les requêtes
            alpha: Poids de la recherche vectorielle (0 = BM25 uniquement, 1 = Chroma uniquement)
            top_k: Nombre de résultats à retourner
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_generator = embedding_generator
        self.alpha = alpha  # Poids pour la fusion
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Recherche hybride : combine Chroma et BM25
        
        Args:
            query: Requête textuelle de l'utilisateur
            top_k: Nombre de résultats (optionnel, utilise self.top_k par défaut)
            
        Returns:
            Liste de résultats fusionnés et classés
        """
        if top_k is None:
            top_k = self.top_k
        
        # --- Étape 1 : Recherche dans Chroma (sémantique) ---
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        chroma_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # --- Étape 2 : Recherche dans BM25 (mots-clés) ---
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        # --- Étape 3 : Fusion des résultats ---
        fused_results = self._fuse_results(chroma_results, bm25_results)
        
        # --- Étape 4 : Trier et retourner top_k ---
        fused_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return fused_results[:top_k]
    
    def _fuse_results(self, chroma_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """
        Fusionne les résultats de Chroma et BM25 en utilisant RRF ou weighted score
        """
        # Créer un dictionnaire pour fusionner par chunk_id
        results_dict = {}
        
        # --- Ajouter les résultats Chroma ---
        for rank, result in enumerate(chroma_results, start=1):
            chunk_id = result['chunk_id']
            
            if chunk_id not in results_dict:
                results_dict[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'chroma_score': result['similarity_score'],
                    'chroma_rank': rank,
                    'bm25_score': 0.0,
                    'bm25_rank': None
                }
            else:
                results_dict[chunk_id]['chroma_score'] = result['similarity_score']
                results_dict[chunk_id]['chroma_rank'] = rank
        
        # --- Ajouter les résultats BM25 ---
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result['chunk_id']
            
            if chunk_id not in results_dict:
                results_dict[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'chroma_score': 0.0,
                    'chroma_rank': None,
                    'bm25_score': result['bm25_score'],
                    'bm25_rank': rank
                }
            else:
                results_dict[chunk_id]['bm25_score'] = result['bm25_score']
                results_dict[chunk_id]['bm25_rank'] = rank
        
        # --- Calculer le score final ---
        fused_results = []
        for chunk_id, result in results_dict.items():
            # Méthode 1 : Reciprocal Rank Fusion (RRF)
            rrf_score = self._calculate_rrf_score(result['chroma_rank'], result['bm25_rank'])
            
            # Méthode 2 : Weighted score (combinaison pondérée)
            weighted_score = self._calculate_weighted_score(
                result['chroma_score'], 
                result['bm25_score']
            )
            
            # Score final = combinaison des deux
            final_score = 0.7 * rrf_score + 0.3 * weighted_score
            
            result['rrf_score'] = rrf_score
            result['weighted_score'] = weighted_score
            result['final_score'] = final_score
            
            fused_results.append(result)
        
        return fused_results
    
    def _calculate_rrf_score(self, chroma_rank: Optional[int], bm25_rank: Optional[int], k: int = 60) -> float:
        """
        Reciprocal Rank Fusion (RRF)
        Score = 1 / (k + rank)
        
        Args:
            chroma_rank: Rang dans les résultats Chroma (None si absent)
            bm25_rank: Rang dans les résultats BM25 (None si absent)
            k: Constante pour éviter la division par zéro (généralement 60)
        """
        rrf_chroma = 1 / (k + chroma_rank) if chroma_rank else 0
        rrf_bm25 = 1 / (k + bm25_rank) if bm25_rank else 0
        
        # Combinaison pondérée
        return self.alpha * rrf_chroma + (1 - self.alpha) * rrf_bm25
    
    def _calculate_weighted_score(self, chroma_score: float, bm25_score: float) -> float:
        """
        Score pondéré : combinaison linéaire des scores normalisés
        """
        # Normaliser les scores (ils sont déjà entre 0 et 1 pour Chroma)
        # Pour BM25, on normalise en divisant par le max possible (approximatif)
        normalized_bm25 = min(bm25_score / 10.0, 1.0)  # BM25 scores souvent < 10
        
        # Combinaison pondérée
        return self.alpha * chroma_score + (1 - self.alpha) * normalized_bm25
    
    def retrieve_with_details(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Variante qui retourne des détails supplémentaires sur la recherche
        """
        if top_k is None:
            top_k = self.top_k
        
        # Recherches séparées
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        chroma_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        # Fusion
        fused_results = self._fuse_results(chroma_results, bm25_results)
        fused_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'query': query,
            'chroma_results': chroma_results[:top_k],
            'bm25_results': bm25_results[:top_k],
            'fused_results': fused_results[:top_k],
            'total_unique_chunks': len(fused_results)
        }
    
    def set_alpha(self, alpha: float):
        """
        Modifie le poids alpha
        alpha=0 : BM25 uniquement
        alpha=0.5 : Équilibre
        alpha=1 : Chroma uniquement
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha doit être entre 0 et 1")
        self.alpha = alpha
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur le retriever"""
        return {
            'vector_store_docs': self.vector_store.collection.count(),
            'bm25_docs': len(self.bm25_index.chunks) if self.bm25_index.chunks else 0,
            'alpha': self.alpha,
            'top_k': self.top_k
        }