"""
Pipeline RAG Financier — point d'entrée dédié aux rapports financiers

Étapes :
  1. Extraction PDF enrichie (texte + tableaux Markdown via pdfplumber)
  2. Nettoyage financier (préserve %, €, $, |, +, …)
  3. Chunking financier (sections financières + tableaux atomiques)
  4. Embeddings (all-MiniLM-L6-v2 par défaut ; FinBERT possible)
  5. Indexation ChromaDB + BM25
  6. Retriever hybride
  7. RAG avec prompt financier
  8. Remplissage d'une table prédéfinie (cas d'usage principal)

Modèles d'embedding recommandés :
  - 'all-MiniLM-L6-v2'          : rapide, polyvalent (défaut)
  - 'all-mpnet-base-v2'          : plus précis, plus lent
  - 'ProsusAI/finbert'           : spécialisé finance (classification)
  - 'yiyanghkust/finbert-tone'   : sentiments financiers
  - 'nickmuchi/financial-roberta-base-embeddings' : embeddings finance
"""

import json
import time
from pathlib import Path

from src.extraction.pdf_extractor import PDFExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.chunker import FinancialChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.rag_pipeline import RAGPipeline
from src.rag.targeted_pipeline import TargetedFinancialPipeline


# ======================================================================
# Configuration — adaptez ces paramètres à votre rapport
# ======================================================================

PDF_PATH = "data/articles/test_article_ARA.pdf"   # Chemin vers le rapport PDF

GEMINI_API_KEY = "AIzaSyDAEAu5gWlmPRVnIc4i5eM2AEZQMph-I3o"  # Clé API Gemini

# Modèle d'embedding — remplacer par un modèle finance si besoin
# Ex: 'nickmuchi/financial-roberta-base-embeddings'
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Table prédéfinie à remplir (exemple : compte de résultat simplifié)
PREDEFINED_TABLE = {
    "table_name": "Compte de résultat simplifié",
    "columns": ["Indicateur", "N", "N-1", "Variation (%)"],
    "rows": [
        "Chiffre d'affaires",
        "EBITDA",
        "EBIT",
        "Résultat net",
        "Marge nette (%)",
    ],
}

# Questions financières de démonstration
TEST_QUESTIONS_FINANCIAL = [
    "Quel est le chiffre d'affaires et son évolution par rapport à l'année précédente ?",
    "Quels sont les principaux risques financiers mentionnés dans le rapport ?",
    "Quelle est la structure de l'endettement et le niveau de trésorerie ?",
]


# ======================================================================
def main():
    print("=" * 65)
    print("  PIPELINE RAG FINANCIER")
    print("=" * 65)

    pdf_path = PDF_PATH
    pdf_name = Path(pdf_path).stem

    # ----------------------------------------------------------------
    # ÉTAPE 1 : Extraction PDF financière
    # ----------------------------------------------------------------
    print("\n📖 ÉTAPE 1/7 : Extraction du rapport financier")
    print("-" * 65)

    extractor = PDFExtractor(pdf_path)

    print("   📄 Extraction texte + tableaux (pdfplumber)…")
    enriched_text = extractor.extract_financial_report()

    # Sauvegarder le texte enrichi (avec tableaux Markdown)
    out_text = f"outputs/extracted_texts/{pdf_name}_financial.txt"
    extractor.save_text(enriched_text, out_text)

    # Sauvegarder les tableaux en JSON structuré séparément
    tables_json = extractor.extract_tables_as_json()
    out_tables = f"outputs/extracted_texts/{pdf_name}_tables.json"
    extractor.save_tables_json(tables_json, out_tables)

    print(f"   ✅ Texte enrichi : {len(enriched_text):,} caractères")
    print(f"   📊 Tableaux JSON extraits : {len(tables_json)}")

    # ----------------------------------------------------------------
    # ÉTAPE 2 : Nettoyage financier
    # ----------------------------------------------------------------
    print("\n🧹 ÉTAPE 2/7 : Nettoyage financier du texte")
    print("-" * 65)

    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_financial(enriched_text)
    print(f"   ✅ Texte nettoyé ({len(cleaned_text):,} caractères)")
    print("      Symboles financiers préservés : %, €, $, |, +, /")

    # ----------------------------------------------------------------
    # ÉTAPE 3 : Chunking financier
    # ----------------------------------------------------------------
    print("\n✂️  ÉTAPE 3/7 : Chunking financier")
    print("-" * 65)

    chunker = FinancialChunker(chunk_size=800, overlap=100, language='fr')
    chunks = chunker.chunk_financial_text(cleaned_text)

    stats = chunker.get_statistics(chunks)
    table_chunks = sum(1 for c in chunks if c['metadata'].get('is_table'))
    narrative_chunks = stats['total_chunks'] - table_chunks

    print(f"   ✅ {stats['total_chunks']} chunks créés")
    print(f"      • Chunks narratifs : {narrative_chunks}")
    print(f"      • Chunks tableau   : {table_chunks} (atomiques, non découpés)")

    out_chunks = f"outputs/chunks/{pdf_name}_financial_chunks.json"
    chunker.save_chunks(chunks, out_chunks)

    # ----------------------------------------------------------------
    # ÉTAPE 4 : Embeddings
    # ----------------------------------------------------------------
    print("\n🔢 ÉTAPE 4/7 : Génération des embeddings")
    print("-" * 65)
    print(f"   Modèle : {EMBEDDING_MODEL}")
    if "finbert" in EMBEDDING_MODEL.lower() or "financial" in EMBEDDING_MODEL.lower():
        print("   ℹ️  Modèle spécialisé finance détecté")

    generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
    chunks_with_embeddings = generator.generate_embeddings(chunks, batch_size=32)

    emb_stats = generator.get_statistics(chunks_with_embeddings)
    print(f"   ✅ {emb_stats['total_embeddings']} embeddings ({emb_stats['embedding_dimension']}d)")

    out_emb = f"outputs/embeddings/{pdf_name}_financial_embeddings.json"
    generator.save_embeddings(chunks_with_embeddings, out_emb)

    # ----------------------------------------------------------------
    # ÉTAPE 5 : Indexation
    # ----------------------------------------------------------------
    print("\n🧱 ÉTAPE 5/7 : Indexation ChromaDB + BM25")
    print("-" * 65)

    vector_store = VectorStore(
        persist_directory=f"outputs/chroma_db/{pdf_name}_financial",
        collection_name=f"{pdf_name}_financial_collection",
    )
    if vector_store.collection.count() > 0:
        vector_store.reset()
    vector_store.add_documents(chunks_with_embeddings)
    print("   ✅ ChromaDB indexé")

    bm25_index = BM25Index()
    bm25_index.index_documents(chunks_with_embeddings)
    out_bm25 = f"outputs/bm25_index/{pdf_name}_financial_bm25.pkl"
    bm25_index.save_index(out_bm25)
    print("   ✅ BM25 indexé")

    # ----------------------------------------------------------------
    # ÉTAPE 6 : Retriever hybride
    # ----------------------------------------------------------------
    print("\n🔍 ÉTAPE 6/7 : Retriever hybride")
    print("-" * 65)

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_generator=generator,
        alpha=0.5,
        top_k=5,
    )
    print("   ✅ Retriever hybride configuré (50% sémantique / 50% BM25)")

    # ----------------------------------------------------------------
    # ÉTAPE 7 : RAG Pipeline financier
    # ----------------------------------------------------------------
    print("\n🤖 ÉTAPE 7/7 : RAG Pipeline avec Gemini (mode financier)")
    print("-" * 65)

    rag = RAGPipeline(
        retriever=retriever,
        llm_provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.2,   # Plus déterministe pour les données chiffrées
        max_tokens=2000,
    )

    # ----------------------------------------------------------------
    # TEST A : Questions financières libres
    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST A — Questions financières libres (mode financial)")
    print("=" * 65)

    for i, question in enumerate(TEST_QUESTIONS_FINANCIAL, 1):
        print(f"\n{'─' * 65}")
        print(f"Question {i}/{len(TEST_QUESTIONS_FINANCIAL)}")
        print(f"❓ {question}")
        print(f"{'─' * 65}")

        result = rag.query(question, top_k=5, return_sources=True, mode="financial")

        print(f"\n💡 RÉPONSE :\n{result['answer']}")
        print(f"\n📚 Sources utilisées : {result['metadata']['num_sources']}")
        for src in result['sources']:
            chunk_type = "(tableau)" if "[TABLE" in src['text_preview'] else "(narratif)"
            print(f"   [{src['source_number']}] {src['chunk_id']} {chunk_type} — score {src['final_score']:.3f}")

        if i < len(TEST_QUESTIONS_FINANCIAL):
            time.sleep(1)

    # ----------------------------------------------------------------
    # TEST B : Remplissage de la table prédéfinie
    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST B — Remplissage de la table prédéfinie")
    print("=" * 65)
    print(f"\n   Table cible : «{PREDEFINED_TABLE['table_name']}»")
    print(f"   Colonnes    : {PREDEFINED_TABLE['columns']}")
    print(f"   Indicateurs : {PREDEFINED_TABLE['rows']}")

    filled = rag.fill_predefined_table(PREDEFINED_TABLE, top_k=5)

    # Affichage Markdown
    md_table = rag.format_filled_table_as_markdown(filled)
    print(f"\n{md_table}")

    # Sauvegarde JSON
    out_table_json = f"outputs/{pdf_name}_filled_table.json"
    Path(out_table_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_table_json, "w", encoding="utf-8") as f:
        json.dump(filled, f, indent=2, ensure_ascii=False)
    print(f"\n   💾 Table sauvegardée : {out_table_json}")

    # Sauvegarde Markdown
    out_table_md = f"outputs/{pdf_name}_filled_table.md"
    with open(out_table_md, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"   💾 Table Markdown     : {out_table_md}")

    # ----------------------------------------------------------------
    # MODE INTERACTIF
    # ----------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("💬 MODE INTERACTIF (financier)")
    print(f"{'=' * 65}")
    print("\nVoulez-vous poser vos propres questions sur le rapport ? (y/n)")

    if input("➤ ").strip().lower() == 'y':
        print("\n✨ Mode chat financier activé ! (tapez 'exit' pour quitter)\n")
        while True:
            question = input("\n❓ Votre question : ").strip()
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Au revoir !")
                break
            if not question:
                continue
            result = rag.query(question, top_k=5, return_sources=False, mode="financial")
            print(f"\n💡 {result['answer']}\n")
            print("─" * 65)

    # ----------------------------------------------------------------
    # RÉSUMÉ FINAL
    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("✅ PIPELINE RAG FINANCIER TERMINÉ")
    print("=" * 65)
    print(f"\n📂 Fichiers générés :")
    print(f"   • Texte enrichi (tableaux Markdown) : {out_text}")
    print(f"   • Tableaux JSON                     : {out_tables}")
    print(f"   • Chunks JSON                       : {out_chunks}")
    print(f"   • Embeddings JSON                   : {out_emb}")
    print(f"   • Base Chroma                       : outputs/chroma_db/{pdf_name}_financial/")
    print(f"   • Index BM25                        : {out_bm25}")
    print(f"   • Table remplie (JSON)              : {out_table_json}")
    print(f"   • Table remplie (Markdown)          : {out_table_md}")
    print(f"\n🎯 Système RAG financier opérationnel !")
    print(f"   • {stats['total_chunks']} chunks ({table_chunks} tableaux atomiques)")
    print(f"   • LLM : Gemini gemini-2.5-flash (mode financier)")
    print(f"   • Symboles financiers préservés dans tout le pipeline\n")
    print("=" * 65 + "\n")


def main_targeted():
    """
    Démo du pipeline ciblé (lazy extraction).

    Différence clé vs main() :
      - Aucune extraction n'est faite au démarrage.
      - Le scan structurel (~50 ms) identifie le plan du document.
      - Chaque requête déclenche l'extraction uniquement des pages pertinentes.
      - Les pages déjà traitées sont mises en cache → latence décroissante.

    Flux d'une requête :
      scan structurel → route query → extract (N pages) → embed → index → RAG
    """
    print("=" * 65)
    print("  PIPELINE RAG FINANCIER — EXTRACTION CIBLÉE (LAZY)")
    print("=" * 65)
    print("  Principe : on extrait UNIQUEMENT les pages pertinentes à")
    print("  chaque requête, identifiées avant toute extraction.")
    print("=" * 65)

    # ── Instanciation du LLM seul (sans pré-extraction) ─────────────────
    print("\n🤖 Initialisation du LLM (sans pré-extraction du PDF)…")
    rag = RAGPipeline(
        retriever=None,          # sera branché dynamiquement
        llm_provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.2,
        max_tokens=2000,
    )

    # ── Création du pipeline ciblé ───────────────────────────────────────
    # Le scan structurel a lieu ici (~50 ms), PAS l'extraction complète
    Path("outputs").mkdir(parents=True, exist_ok=True)
    pipeline = TargetedFinancialPipeline(
        pdf_path=PDF_PATH,
        rag_pipeline=rag,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=800,
        overlap=100,
        language="fr",
        top_k=5,
        output_dir="outputs",
    )

    # ── TEST A : Questions ciblées ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST A — Requêtes ciblées (extraction lazy par requête)")
    print("=" * 65)

    for i, question in enumerate(TEST_QUESTIONS_FINANCIAL, 1):
        print(f"\n{'─' * 65}")
        print(f"Question {i}/{len(TEST_QUESTIONS_FINANCIAL)}")
        print(f"❓ {question}")

        result = pipeline.query(question, mode="financial")

        print(f"\n💡 RÉPONSE :\n{result['answer']}")
        print(f"\n   ⏱  Routage : {result['routing_ms']} ms | "
              f"Total : {result['total_ms']} ms | "
              f"Pages ciblées : {len(result['targeted_pages'])}")

        stats = pipeline.get_cache_stats()
        print(f"   📊 Cache : {stats['indexed_pages']}/{stats['total_pages']} pages "
              f"({stats['coverage_pct']} %) — {stats['total_chunks']} chunks")

        if i < len(TEST_QUESTIONS_FINANCIAL):
            time.sleep(1)

    # ── TEST B : Table prédéfinie avec ciblage par indicateur ────────────
    print("\n" + "=" * 65)
    print("TEST B — Table prédéfinie (extraction ciblée par indicateur)")
    print("=" * 65)

    filled = pipeline.fill_predefined_table(PREDEFINED_TABLE)
    md_table = rag.format_filled_table_as_markdown(filled)
    print(f"\n{md_table}")

    out_table = f"outputs/{Path(PDF_PATH).stem}_filled_targeted.md"
    with open(out_table, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"\n   💾 Table sauvegardée : {out_table}")

    # ── Résumé final ─────────────────────────────────────────────────────
    stats = pipeline.get_cache_stats()
    print("\n" + "=" * 65)
    print("✅ PIPELINE CIBLÉ TERMINÉ")
    print("=" * 65)
    print(f"\n   Pages extraites  : {stats['indexed_pages']} / {stats['total_pages']} "
          f"({stats['coverage_pct']} % du document)")
    print(f"   Chunks indexés   : {stats['total_chunks']}")
    print(f"   Économie         : {100 - stats['coverage_pct']:.1f} % du document non extrait\n")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys
    if "--targeted" in sys.argv:
        main_targeted()
    else:
        main()
