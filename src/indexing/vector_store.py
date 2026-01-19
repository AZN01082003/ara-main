"""
Gestion de la base de données vectorielle (Chroma)
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict
import numpy as np


class VectorStore:
    """
    Gère la base de données vectorielle avec ChromaDB
    """
    
    def __init__(self, persist_directory="outputs/chroma_db", collection_name="scientific_papers"):
        """
        Args:
            persist_directory: Dossier où sauvegarder la base de données
            collection_name: Nom de la collection Chroma
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialiser le client Chroma
        print(f"   📦 Initialisation de ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Créer ou récupérer la collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"   ✅ Collection '{collection_name}' chargée ({self.collection.count()} documents)")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Utiliser la similarité cosinus
            )
            print(f"   ✅ Nouvelle collection '{collection_name}' créée")
    
    def add_documents(self, chunks_with_embeddings: List[Dict]):
        """
        Ajoute des documents avec leurs embeddings à la collection
        
        Args:
            chunks_with_embeddings: Liste de chunks avec embeddings
        """
        print(f"   🔄 Indexation de {len(chunks_with_embeddings)} documents dans Chroma...")
        
        # Préparer les données pour Chroma
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks_with_embeddings:
            ids.append(chunk['chunk_id'])
            documents.append(chunk['text'])
            embeddings.append(chunk['embedding'])
            
            # Métadonnées (sans l'embedding pour éviter la redondance)
            metadata = {
                'length': chunk['length'],
                'word_count': chunk['word_count'],
                'position': chunk['metadata']['position'],
                'total_chunks': chunk['metadata']['total_chunks']
            }
            metadatas.append(metadata)
        
        # Ajouter à Chroma (batch)
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"   ✅ {len(chunks_with_embeddings)} documents indexés dans Chroma")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Recherche les documents les plus similaires
        
        Args:
            query_embedding: Embedding de la requête
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de résultats avec documents, scores et métadonnées
        """
        # Convertir en liste si nécessaire
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Recherche dans Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Formater les résultats
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity_score': 1 - results['distances'][0][i]  # Convertir distance en similarité
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Retourne des statistiques sur la collection"""
        return {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count(),
            'persist_directory': str(self.persist_directory)
        }
    
    def delete_collection(self):
        """Supprime la collection (utile pour reset)"""
        self.client.delete_collection(name=self.collection_name)
        print(f"   🗑️ Collection '{self.collection_name}' supprimée")
    
    def reset(self):
        """Reset complet : supprime et recrée la collection"""
        try:
            self.delete_collection()
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"   ♻️ Collection '{self.collection_name}' réinitialisée")