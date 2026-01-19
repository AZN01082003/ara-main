"""
Pipeline RAG COMPLET - Étapes 1 à 6
Extraction PDF + Nettoyage + Chunking + Embeddings + Indexation + Retrieval + RAG
Avec Google Gemini (gemini-1.5-flash)
"""

from pathlib import Path
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.ocr_handler import OCRHandler
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.rag_pipeline import RAGPipeline


def main():
    print("🚀 Démarrage du pipeline RAG COMPLET\n")
    print("="*60)
    
    # ============================================================
    # ÉTAPE 1 : EXTRACTION DU PDF
    # ============================================================
    print("\n📖 ÉTAPE 1/6 : Extraction du PDF")
    print("-"*60)
    
    pdf_path = "data/articles/test_article_ARA.pdf"
    pdf_name = Path(pdf_path).stem
    
    extractor = PDFExtractor(pdf_path)
    print("   📄 Extraction en cours avec PyMuPDF...")
    text = extractor.extract_with_pymupdf()
    
    output_text_path = f"outputs/extracted_texts/{pdf_name}.txt"
    extractor.save_text(text, output_text_path)
    
    print(f"   ✅ Extraction terminée !")
    print(f"   📄 Nombre de caractères : {len(text):,}")
    print(f"   📄 Nombre de mots : {len(text.split()):,}")
    
    # ============================================================
    # ÉTAPE 2 : NETTOYAGE ET CHUNKING
    # ============================================================
    print("\n✂️ ÉTAPE 2/6 : Nettoyage et Chunking")
    print("-"*60)
    
    print("   🧹 Nettoyage du texte...")
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(text)
    print(f"   ✅ Texte nettoyé ({len(cleaned_text):,} caractères)")
    
    print("\n   ✂️ Découpage en chunks...")
    chunker = SemanticChunker(chunk_size=500, overlap=50, language='en')
    chunks = chunker.chunk_text(cleaned_text)
    
    stats = chunker.get_statistics(chunks)
    print(f"   ✅ {stats['total_chunks']} chunks créés")
    
    output_chunks_path = f"outputs/chunks/{pdf_name}_chunks.json"
    chunker.save_chunks(chunks, output_chunks_path)
    
    # ============================================================
    # ÉTAPE 3 : GÉNÉRATION DES EMBEDDINGS
    # ============================================================
    print("\n🔢 ÉTAPE 3/6 : Génération des embeddings")
    print("-"*60)
    
    generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    chunks_with_embeddings = generator.generate_embeddings(chunks, batch_size=32)
    
    emb_stats = generator.get_statistics(chunks_with_embeddings)
    print(f"   ✅ {emb_stats['total_embeddings']} embeddings générés")
    
    output_embeddings_path = f"outputs/embeddings/{pdf_name}_embeddings.json"
    generator.save_embeddings(chunks_with_embeddings, output_embeddings_path)
    
    # ============================================================
    # ÉTAPE 4 : INDEXATION
    # ============================================================
    print("\n🧱 ÉTAPE 4/6 : Indexation dans les bases de données")
    print("-"*60)
    
    print("   📦 Indexation dans ChromaDB...")
    vector_store = VectorStore(
        persist_directory=f"outputs/chroma_db/{pdf_name}",
        collection_name=f"{pdf_name}_collection"
    )
    
    if vector_store.collection.count() > 0:
        vector_store.reset()
    
    vector_store.add_documents(chunks_with_embeddings)
    print(f"   ✅ ChromaDB indexé")
    
    print("\n   🔍 Indexation dans BM25...")
    bm25_index = BM25Index()
    bm25_index.index_documents(chunks_with_embeddings)
    
    bm25_path = f"outputs/bm25_index/{pdf_name}_bm25.pkl"
    bm25_index.save_index(bm25_path)
    print(f"   ✅ BM25 indexé")
    
    # ============================================================
    # ÉTAPE 5 : CONFIGURATION DU RETRIEVER
    # ============================================================
    print("\n🔍 ÉTAPE 5/6 : Configuration du Retriever Hybride")
    print("-"*60)
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_generator=generator,
        alpha=0.5,
        top_k=5
    )
    print("   ✅ Retriever hybride configuré")
    
    # ============================================================
    # ÉTAPE 6 : CONFIGURATION DU RAG PIPELINE AVEC GEMINI
    # ============================================================
    print("\n🤖 ÉTAPE 6/6 : Configuration du système RAG avec Gemini")
    print("-"*60)
    
    print("   ⚙️  Initialisation du RAG Pipeline avec Google Gemini...")
    
    # Configuration avec Gemini
    # Remplacez "YOUR_GEMINI_API_KEY" par votre vraie clé API
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider="gemini",
        model_name="gemini-2.5-flash",  # ou "gemini-1.5-pro" pour plus de qualité
        api_key="AIzaSyDAEAu5gWlmPRVnIc4i5eM2AEZQMph-I3o",  # ⚠️ REMPLACEZ PAR VOTRE CLÉ API
        temperature=0.7,
        max_tokens=1000
    )
    
    rag_stats = rag_pipeline.get_statistics()
    print(f"   📊 Configuration RAG :")
    print(f"      • LLM Provider : {rag_stats['llm_provider']}")
    print(f"      • Modèle : {rag_stats['model_name']}")
    print(f"      • Temperature : {rag_stats['temperature']}")
    print(f"      • Max tokens : {rag_stats['max_tokens']}")
    
    # ============================================================
    # TEST COMPLET : Questions-Réponses avec RAG
    # ============================================================
    print("\n🧪 TEST : Système RAG complet (Q&A)")
    print("="*60)
    
    # Liste de questions test
    test_questions = [
        "What is the main methodology used in this paper?",
        "What are the key findings or results?",
        "What are the limitations mentioned in the paper?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(test_questions)}")
        print(f"{'='*60}")
        print(f"\n❓ {question}")
        print(f"\n{'─'*60}")
        
        # Exécuter la requête RAG
        result = rag_pipeline.query(question, top_k=3, return_sources=True)
        
        # Afficher la réponse
        print(f"💡 RÉPONSE :")
        print(f"{result['answer']}")
        
        # Afficher les sources
        print(f"\n📚 SOURCES UTILISÉES ({result['metadata']['num_sources']}) :")
        for source in result['sources']:
            print(f"\n   [{source['source_number']}] Chunk: {source['chunk_id']}")
            print(f"       Score: {source['final_score']:.3f}")
            print(f"       Extrait: {source['text_preview'][:100]}...")
        
        # Petite pause entre les questions (optionnelle avec Gemini)
        import time
        if i < len(test_questions):
            print("\n   ⏳ Pause de 1 seconde...")
            time.sleep(1)
    
    # ============================================================
    # MODE INTERACTIF
    # ============================================================
    print(f"\n{'='*60}")
    print("💬 MODE INTERACTIF")
    print(f"{'='*60}")
    print("\n✨ Gemini est rapide et a des limites généreuses !")
    print("Voulez-vous poser vos propres questions ? (y/n)")
    
    user_input = input("➤ ").strip().lower()
    
    if user_input == 'y':
        print("\n✨ Mode chat activé ! (tapez 'exit' pour quitter)\n")
        
        while True:
            question = input("\n❓ Votre question : ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Au revoir !")
                break
            
            if not question:
                continue
            
            print("\n🤔 Recherche et génération en cours...")
            
            result = rag_pipeline.query(question, top_k=3, return_sources=True)
            
            print(f"\n💡 RÉPONSE :")
            print(f"{result['answer']}\n")
            
            print(f"📚 Sources : {result['metadata']['num_sources']} documents utilisés")
            print("-"*60)
    
    # ============================================================
    # RÉSUMÉ FINAL
    # ============================================================
    print("\n" + "="*60)
    print("✅ PIPELINE RAG COMPLET TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print(f"\n📂 Tous les fichiers générés :")
    print(f"   • Texte brut       : {output_text_path}")
    print(f"   • Chunks JSON      : {output_chunks_path}")
    print(f"   • Embeddings JSON  : {output_embeddings_path}")
    print(f"   • Base Chroma      : outputs/chroma_db/{pdf_name}/")
    print(f"   • Index BM25       : {bm25_path}")
    
    print(f"\n🎯 Système RAG opérationnel !")
    print(f"   • Documents indexés : {stats['total_chunks']}")
    print(f"   • LLM : Google Gemini ({rag_stats['model_name']})")
    print(f"   • Prêt pour l'analyse d'articles scientifiques ! 🚀")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()