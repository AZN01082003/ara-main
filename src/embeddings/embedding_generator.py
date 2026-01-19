"""
Génération d'embeddings pour les chunks de texte
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Génère des embeddings (vecteurs) pour les chunks de texte
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Args:
            model_name: Nom du modèle à utiliser
            
        Modèles recommandés :
        - 'all-MiniLM-L6-v2' : Rapide, léger (384 dimensions)
        - 'all-mpnet-base-v2' : Plus précis (768 dimensions)
        - 'allenai/specter2' : Spécialisé pour articles scientifiques
        """
        print(f"   📥 Chargement du modèle : {model_name}...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   ✅ Modèle chargé ({self.embedding_dim} dimensions)")
    
    def generate_embeddings(self, chunks: List[Dict], batch_size=32) -> List[Dict]:
        """
        Génère les embeddings pour une liste de chunks
        
        Args:
            chunks: Liste de dictionnaires avec clé 'text'
            batch_size: Nombre de chunks à traiter en même temps
            
        Returns:
            Liste de chunks avec embeddings ajoutés
        """
        print(f"   🔄 Génération de {len(chunks)} embeddings...")
        
        # Extraire les textes
        texts = [chunk['text'] for chunk in chunks]
        
        # Générer les embeddings (avec barre de progression)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Ajouter les embeddings aux chunks
        chunks_with_embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embeddings[i].tolist()  # Convertir en liste pour JSON
            chunk_copy['embedding_model'] = self.model_name
            chunk_copy['embedding_dim'] = self.embedding_dim
            chunks_with_embeddings.append(chunk_copy)
        
        print(f"   ✅ Embeddings générés !")
        
        return chunks_with_embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Génère l'embedding pour un seul texte
        (utile pour les requêtes utilisateur)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def save_embeddings(self, chunks_with_embeddings: List[Dict], output_path: str):
        """Sauvegarde les chunks avec leurs embeddings en JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Embeddings sauvegardés : {output_path}")
    
    def load_embeddings(self, input_path: str) -> List[Dict]:
        """Charge les chunks avec embeddings depuis JSON"""
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Reconvertir les embeddings en numpy arrays
        for chunk in chunks:
            if 'embedding' in chunk:
                chunk['embedding'] = np.array(chunk['embedding'])
        
        return chunks
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings
        Retourne un score entre -1 et 1 (1 = très similaire)
        """
        from numpy.linalg import norm
        
        # Similarité cosinus
        similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         chunks_with_embeddings: List[Dict], 
                         top_k: int = 5) -> List[Dict]:
        """
        Trouve les chunks les plus similaires à une requête
        
        Args:
            query_embedding: Embedding de la requête
            chunks_with_embeddings: Liste des chunks avec embeddings
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste des top_k chunks les plus similaires avec leur score
        """
        # Calculer les similarités
        similarities = []
        for chunk in chunks_with_embeddings:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append({
                'chunk': chunk,
                'similarity_score': similarity
            })
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Retourner les top_k
        return similarities[:top_k]
    
    def get_statistics(self, chunks_with_embeddings: List[Dict]) -> Dict:
        """Calcule des statistiques sur les embeddings"""
        embeddings = [np.array(chunk['embedding']) for chunk in chunks_with_embeddings]
        embeddings_matrix = np.array(embeddings)
        
        return {
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings_matrix.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings_matrix, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings_matrix, axis=1))),
            'model_name': self.model_name
        }