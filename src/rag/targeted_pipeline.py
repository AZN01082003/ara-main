"""
Pipeline RAG financier ciblé (lazy extraction).

Principe : on n'extrait, on ne nettoie, on ne chunk, on n'embed que les
pages réellement pertinentes à la requête — identifiées AVANT toute
extraction via un scan structurel ultra-rapide du PDF.

Architecture :

  PDF
   │
   ├─[1] PDFStructureScanner.scan()          ← quasi-instantané (~50 ms)
   │      → StructuralMap {sections, pages}
   │
   ├─[2] SectionRouter.route(query)           ← instantané (mots-clés)
   │      → sections cibles + plages de pages
   │
   ├─[3] PDFExtractor (pages ciblées seulement)  ← 5-20 pages au lieu de 300+
   │      → texte + tableaux Markdown
   │
   ├─[4] TextCleaner.clean_financial()
   │
   ├─[5] FinancialChunker.chunk_financial_text()
   │
   ├─[6] EmbeddingGenerator  (sur N chunks, pas tout le doc)
   │
   ├─[7] VectorStore + BM25Index  (index partiel, par section)
   │
   └─[8] RAGPipeline.query(mode='financial')

Cache intégré : les sections déjà extraites et indexées sont mises en
cache pour éviter de re-traiter lors d'une deuxième requête sur le même
périmètre.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.pdf_structure_scanner import PDFStructureScanner
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.chunker import FinancialChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.rag_pipeline import RAGPipeline
from src.routing.section_router import SectionRouter


class TargetedFinancialPipeline:
    """
    Pipeline RAG financier avec extraction ciblée (lazy).

    Workflow par requête :
      scan structural → route query → extract targeted pages →
      clean → chunk → embed → index → retrieve → LLM answer

    Le scan structurel n'est fait qu'une seule fois (au __init__).
    Le cache de sections évite de re-traiter les pages déjà indexées.
    """

    def __init__(
        self,
        pdf_path: str,
        rag_pipeline: RAGPipeline,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        overlap: int = 100,
        language: str = "fr",
        top_k: int = 5,
        output_dir: str = "outputs",
    ):
        """
        Args:
            pdf_path: Chemin vers le PDF financier.
            rag_pipeline: Instance RAGPipeline déjà configurée avec un LLM.
            embedding_model: Modèle d'embedding à utiliser.
            chunk_size: Taille des chunks narratifs.
            overlap: Chevauchement entre chunks.
            language: 'fr' ou 'en'.
            top_k: Nombre de chunks à récupérer par requête RAG.
            output_dir: Dossier pour les fichiers intermédiaires.
        """
        self.pdf_path = Path(pdf_path)
        self.pdf_name = self.pdf_path.stem
        self.rag = rag_pipeline
        self.top_k = top_k
        self.output_dir = Path(output_dir)

        # Composants du pipeline
        self.extractor = PDFExtractor(pdf_path)
        self.cleaner = TextCleaner()
        self.chunker = FinancialChunker(chunk_size=chunk_size, overlap=overlap, language=language)
        self.generator = EmbeddingGenerator(model_name=embedding_model)

        # Cache : pages déjà extraites et indexées → évite re-processing
        self._indexed_page_ranges: List[tuple] = []   # [(start, end), …]
        self._all_chunks: List[Dict] = []

        # Index partagé (alimenté progressivement)
        self._vector_store: Optional[VectorStore] = None
        self._bm25_index: Optional[BM25Index] = None
        self._retriever: Optional[HybridRetriever] = None

        # ── Étape 1 : scan structurel (une seule fois) ──────────────────
        t0 = time.time()
        scanner = PDFStructureScanner(pdf_path)
        self.structural_map = scanner.scan()
        self.router = SectionRouter(self.structural_map)
        dt = time.time() - t0
        scanner.print_structural_map(self.structural_map)
        print(f"   ⚡ Scan structurel terminé en {dt*1000:.0f} ms")

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def query(self, question: str, mode: str = "financial") -> Dict:
        """
        Répond à une question financière avec extraction ciblée.

        Args:
            question: Question de l'utilisateur.
            mode: "financial" (défaut) ou "standard".

        Returns:
            Résultat RAG avec 'answer', 'sources', 'targeted_pages'.
        """
        t_total = time.time()

        # ── Étape 2 : routage ───────────────────────────────────────────
        t0 = time.time()
        target_pages = self.router.pages_for_query(question)
        dt_route = time.time() - t0

        self.router.print_routing(question)

        if not target_pages:
            # Fallback : tout le document
            target_pages = list(range(1, self.structural_map["total_pages"] + 1))
            print(f"   ⚠️  Aucune section ciblée — traitement du document complet")
        else:
            print(f"   📄 Pages ciblées : {target_pages[0]}–{target_pages[-1]} "
                  f"({len(target_pages)} pages sur {self.structural_map['total_pages']})")

        # ── Étapes 3-6 : extract / clean / chunk / embed (si nouvelles pages) ──
        new_pages = self._filter_new_pages(target_pages)
        if new_pages:
            t0 = time.time()
            new_chunks = self._process_pages(new_pages)
            dt_proc = time.time() - t0
            print(f"   ✅ {len(new_chunks)} chunks traités en {dt_proc*1000:.0f} ms "
                  f"({len(new_pages)} nouvelles pages)")

            # ── Étape 7 : indexation incrémentale ───────────────────────
            self._update_index(new_chunks)
            self._mark_pages_indexed(new_pages)
        else:
            print(f"   ♻️  Pages déjà indexées — extraction ignorée (cache hit)")

        # ── Étape 8 : RAG ───────────────────────────────────────────────
        # Brancher le retriever courant dans le pipeline RAG
        self.rag.retriever = self._retriever

        result = self.rag.query(question, top_k=self.top_k,
                                return_sources=True, mode=mode)
        result["targeted_pages"] = target_pages
        result["routing_ms"] = round(dt_route * 1000)
        result["total_ms"] = round((time.time() - t_total) * 1000)

        return result

    def fill_predefined_table(self, table_template: Dict) -> Dict:
        """
        Remplit une table prédéfinie avec extraction ciblée par indicateur.

        Pour chaque ligne de la table, le router identifie les pages
        pertinentes et extrait seulement celles-ci avant de lancer le RAG.

        Args:
            table_template: {"table_name", "columns", "rows"}

        Returns:
            Table remplie (même format que RAGPipeline.fill_predefined_table)
        """
        table_name = table_template.get("table_name", "Tableau")
        columns = table_template.get("columns", [])
        rows = table_template.get("rows", [])
        data_columns = columns[1:] if len(columns) > 1 else columns

        print(f"\n   📋 Remplissage ciblé de : «{table_name}»")

        # Pré-router toutes les lignes → collecter les pages à extraire
        routing_map = self.router.route_table_template(table_template)
        all_pages: set = set()
        for row, sections in routing_map.items():
            for s in sections:
                for p in range(s["start_page"], s["end_page"] + 1):
                    all_pages.add(p)

        # Extraire toutes les pages nécessaires en une seule passe
        new_pages = self._filter_new_pages(sorted(all_pages))
        if new_pages:
            new_chunks = self._process_pages(new_pages)
            self._update_index(new_chunks)
            self._mark_pages_indexed(new_pages)
            print(f"   ✅ {len(new_chunks)} chunks extraits sur {len(new_pages)} pages ciblées")
        else:
            print(f"   ♻️  Toutes les pages déjà en cache")

        # Brancher le retriever
        self.rag.retriever = self._retriever

        # Remplir la table via RAGPipeline standard
        filled = self.rag.fill_predefined_table(table_template, top_k=self.top_k)
        return filled

    # ------------------------------------------------------------------
    # Extraction ciblée (pages spécifiques)
    # ------------------------------------------------------------------

    def _process_pages(self, pages: List[int]) -> List[Dict]:
        """
        Extrait, nettoie, découpe et embed un sous-ensemble de pages.

        Args:
            pages: Numéros de pages 1-indexed à traiter.

        Returns:
            Chunks avec embeddings.
        """
        # Extraction ciblée via pdfplumber
        raw_text = self._extract_pages(pages)

        # Nettoyage financier
        cleaned = self.cleaner.clean_financial(raw_text)

        # Chunking financier (tableaux atomiques)
        chunks = self.chunker.chunk_financial_text(cleaned)

        if not chunks:
            return []

        # Ajouter numéros de pages aux métadonnées
        page_tag = f"p{pages[0]}-{pages[-1]}"
        for chunk in chunks:
            chunk["metadata"]["source_pages"] = pages
            chunk["chunk_id"] = f"{page_tag}_{chunk['chunk_id']}"

        # Embeddings
        chunks_with_emb = self.generator.generate_embeddings(chunks, batch_size=32)

        # Mise à jour de la liste globale
        self._all_chunks.extend(chunks_with_emb)

        return chunks_with_emb

    def _extract_pages(self, pages: List[int]) -> str:
        """
        Extrait texte + tableaux Markdown pour une liste de numéros de pages.
        """
        parts = []
        page_set = set(pages)

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if (page_num + 1) not in page_set:
                    continue

                page_parts = [f"\n--- Page {page_num + 1} ---\n"]

                # Texte narratif
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_parts.append(page_text)

                # Tableaux → Markdown structuré
                page_tables = page.extract_tables()
                if page_tables:
                    for t_idx, table in enumerate(page_tables):
                        md = self.extractor._table_to_markdown(
                            table, page_num + 1, t_idx + 1
                        )
                        if md:
                            page_parts.append(md)

                parts.append("\n".join(page_parts))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Indexation incrémentale
    # ------------------------------------------------------------------

    def _update_index(self, new_chunks: List[Dict]):
        """Ajoute les nouveaux chunks à l'index existant (ou le crée)."""
        if not new_chunks:
            return

        collection_name = f"{self.pdf_name}_targeted"
        chroma_dir = str(self.output_dir / "chroma_db" / collection_name)

        if self._vector_store is None:
            self._vector_store = VectorStore(
                persist_directory=chroma_dir,
                collection_name=collection_name,
            )

        self._vector_store.add_documents(new_chunks)

        if self._bm25_index is None:
            self._bm25_index = BM25Index()
        self._bm25_index.index_documents(self._all_chunks)  # re-index complet (BM25 stateless)

        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            bm25_index=self._bm25_index,
            embedding_generator=self.generator,
            alpha=0.5,
            top_k=self.top_k,
        )

    # ------------------------------------------------------------------
    # Gestion du cache de pages
    # ------------------------------------------------------------------

    def _filter_new_pages(self, pages: List[int]) -> List[int]:
        """Retourne uniquement les pages pas encore indexées."""
        already_indexed = set()
        for start, end in self._indexed_page_ranges:
            for p in range(start, end + 1):
                already_indexed.add(p)
        return [p for p in pages if p not in already_indexed]

    def _mark_pages_indexed(self, pages: List[int]):
        """Enregistre les pages comme indexées dans le cache."""
        if pages:
            self._indexed_page_ranges.append((min(pages), max(pages)))

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> Dict:
        """Retourne les statistiques du cache d'extraction."""
        indexed = sum(e - s + 1 for s, e in self._indexed_page_ranges)
        return {
            "total_pages": self.structural_map["total_pages"],
            "indexed_pages": indexed,
            "coverage_pct": round(100 * indexed / max(1, self.structural_map["total_pages"]), 1),
            "total_chunks": len(self._all_chunks),
            "cached_ranges": self._indexed_page_ranges,
        }
