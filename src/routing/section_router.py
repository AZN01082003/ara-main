"""
Routeur sémantique : mappe une requête utilisateur vers les sections
pertinentes du rapport financier AVANT d'extraire quoi que ce soit.

Deux niveaux de routage (par ordre de priorité croissante) :
  1. Routage par mots-clés (instantané, aucun LLM)
  2. Routage assisté LLM  (si ambigu ou si LLM disponible)

Utilisation typique :
    router = SectionRouter(structural_map)
    target_sections = router.route("Quel est l'EBITDA 2023 ?")
    # → [{"title": "Compte de résultat", "start_page": 45, "end_page": 52}, …]
"""

import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dictionnaire de correspondance financière : terme → sections cibles
# (liste ordonnée par priorité)
# ---------------------------------------------------------------------------
FINANCIAL_KEYWORD_MAP: List[Dict] = [
    # ── Compte de résultat ──────────────────────────────────────────────────
    {
        "keywords": [
            "chiffre d'affaires", "ca", "revenus", "revenue",
            "ebitda", "ebit", "résultat opérationnel", "operating income",
            "résultat net", "net income", "marge", "margin",
            "charges", "coûts", "dépenses", "expenses",
            "résultat", "bénéfice", "perte", "profit", "loss",
            "croissance", "growth",
        ],
        "sections": ["compte de résultat", "income statement", "résultats",
                     "performance", "chiffre d'affaires"],
    },
    # ── Bilan ────────────────────────────────────────────────────────────────
    {
        "keywords": [
            "bilan", "balance sheet", "actif", "passif", "asset", "liability",
            "capitaux propres", "equity", "fonds propres",
            "immobilisations", "stocks", "créances", "dettes",
            "goodwill", "écarts d'acquisition",
        ],
        "sections": ["bilan", "balance sheet", "état financier",
                     "capitaux propres", "equity"],
    },
    # ── Flux de trésorerie ───────────────────────────────────────────────────
    {
        "keywords": [
            "trésorerie", "cash", "flux", "cash flow", "liquidité",
            "free cash flow", "fcf", "investissement", "capex",
            "financement", "remboursement", "dette nette", "net debt",
        ],
        "sections": ["flux de trésorerie", "cash flow", "trésorerie",
                     "tableau des flux"],
    },
    # ── Endettement ──────────────────────────────────────────────────────────
    {
        "keywords": [
            "dette", "debt", "endettement", "emprunt", "obligation",
            "leverage", "levier", "ratio dette", "covenant",
            "crédit", "ligne de crédit", "refinancement",
        ],
        "sections": ["endettement", "dette", "financement", "notes annexes",
                     "bilan"],
    },
    # ── Dividendes ───────────────────────────────────────────────────────────
    {
        "keywords": [
            "dividende", "dividend", "distribution", "coupon",
            "rendement", "yield", "rachat d'actions", "buyback",
        ],
        "sections": ["dividendes", "capitaux propres", "résultats"],
    },
    # ── Risques ──────────────────────────────────────────────────────────────
    {
        "keywords": [
            "risque", "risk", "incertitude", "exposition",
            "couverture", "hedge", "sensibilité", "stress test",
            "litiges", "contentieux",
        ],
        "sections": ["risques", "risk factors", "notes annexes", "gouvernance"],
    },
    # ── Gouvernance / ESG ────────────────────────────────────────────────────
    {
        "keywords": [
            "gouvernance", "governance", "conseil d'administration", "board",
            "dirigeants", "management", "rémunération", "compensation",
            "esg", "environnement", "social", "durabilité", "sustainability",
        ],
        "sections": ["gouvernance", "rapport de gestion", "faits marquants"],
    },
    # ── Perspectives ─────────────────────────────────────────────────────────
    {
        "keywords": [
            "perspectives", "outlook", "guidance", "prévision", "forecast",
            "objectif", "target", "ambition", "stratégie", "strategy",
        ],
        "sections": ["perspectives", "rapport de gestion", "faits marquants"],
    },
]


class SectionRouter:
    """
    Route une requête utilisateur vers les sections pertinentes du document.

    Utilise d'abord la correspondance par mots-clés (O(1)), puis
    optionnellement un LLM pour les requêtes ambiguës.
    """

    def __init__(
        self,
        structural_map: Dict,
        llm_client=None,         # Instance optionnelle du LLM (Gemini, OpenAI…)
        llm_call_fn=None,        # Callable(prompt) → str, pour rester agnostique au provider
        max_sections: int = 3,   # Nombre maximum de sections à cibler
    ):
        """
        Args:
            structural_map: Résultat de PDFStructureScanner.scan()
            llm_client: Client LLM optionnel (pour le routage assisté)
            llm_call_fn: Fonction callable(prompt: str) -> str pour appel LLM
            max_sections: Limite le nombre de sections renvoyées
        """
        self.structural_map = structural_map
        self.sections = structural_map.get("sections", [])
        self.total_pages = structural_map.get("total_pages", 1)
        self.llm_call_fn = llm_call_fn
        self.max_sections = max_sections

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------

    def route(self, query: str) -> List[Dict]:
        """
        Mappe la requête vers les sections pertinentes.

        Returns:
            Liste de sections triées par pertinence :
            [{"title", "start_page", "end_page", "level", "score"}, …]
        """
        query_lower = query.lower()

        # Étape 1 : correspondance par mots-clés
        keyword_results = self._keyword_route(query_lower)

        # Étape 2 : si pas de résultat clair ET LLM disponible → routage LLM
        if not keyword_results and self.llm_call_fn:
            llm_results = self._llm_route(query)
            return llm_results[:self.max_sections]

        # Étape 3 : si toujours rien → renvoyer les premières sections (fallback)
        if not keyword_results:
            return self._fallback_sections()

        return keyword_results[:self.max_sections]

    def route_table_template(self, table_template: Dict) -> Dict[str, List[Dict]]:
        """
        Route une table prédéfinie : associe chaque indicateur (ligne)
        à ses sections sources.

        Args:
            table_template: {"rows": ["EBITDA", "CA", …], …}

        Returns:
            {row_name: [sections…], …}
        """
        routing_map = {}
        for row in table_template.get("rows", []):
            routing_map[row] = self.route(row)
        return routing_map

    # ------------------------------------------------------------------
    # Routage par mots-clés
    # ------------------------------------------------------------------

    def _keyword_route(self, query_lower: str) -> List[Dict]:
        """
        Cherche les mots-clés de la requête dans le dictionnaire financier,
        puis matche les sections cibles contre le plan du document.
        """
        # Compter les hits par groupe de sections cibles
        group_scores: Dict[int, float] = {}
        group_targets: Dict[int, List[str]] = {}

        for i, group in enumerate(FINANCIAL_KEYWORD_MAP):
            score = sum(
                1 + 0.5 * (kw in query_lower)   # bonus si le kw exact est présent
                for kw in group["keywords"]
                if kw in query_lower
            )
            if score > 0:
                group_scores[i] = score
                group_targets[i] = group["sections"]

        if not group_scores:
            return []

        # Agréger les sections cibles pondérées par leur score
        section_scores: Dict[str, float] = {}
        for i, score in group_scores.items():
            for target in group_targets[i]:
                section_scores[target] = section_scores.get(target, 0) + score

        # Matcher contre les sections réelles du document
        matched = self._match_sections(section_scores)
        return matched

    def _match_sections(self, section_scores: Dict[str, float]) -> List[Dict]:
        """
        Mappe les noms de sections cibles aux entrées réelles du structural_map.
        Utilise une correspondance approximative (substring, case-insensitive).
        """
        results: Dict[str, Dict] = {}   # keyed by section title to deduplicate

        for target_name, score in section_scores.items():
            for section in self.sections:
                s_title = section["title"].lower()
                t_lower = target_name.lower()

                if t_lower in s_title or s_title in t_lower:
                    key = section["title"]
                    if key not in results or results[key]["score"] < score:
                        results[key] = {**section, "score": score}

        # Trier par score décroissant
        return sorted(results.values(), key=lambda x: -x["score"])

    # ------------------------------------------------------------------
    # Routage assisté LLM
    # ------------------------------------------------------------------

    def _llm_route(self, query: str) -> List[Dict]:
        """
        Demande au LLM quelles sections du plan sont pertinentes pour la requête.
        Appelé uniquement si le routage par mots-clés échoue.
        """
        sections_list = "\n".join(
            f"  - [{s['start_page']}-{s['end_page']}] {s['title']}"
            for s in self.sections
        )

        prompt = (
            f"Voici le plan d'un rapport financier :\n{sections_list}\n\n"
            f"Question de l'utilisateur : «{query}»\n\n"
            f"Quelles sections contiennent vraisemblablement la réponse ? "
            f"Réponds UNIQUEMENT avec les titres exacts des sections pertinentes, "
            f"séparés par des virgules, maximum {self.max_sections} sections."
        )

        try:
            raw = self.llm_call_fn(prompt)
            titles = [t.strip() for t in raw.split(",") if t.strip()]
            matched = []
            for title in titles:
                for section in self.sections:
                    if title.lower() in section["title"].lower():
                        matched.append({**section, "score": 1.0})
                        break
            return matched
        except Exception:
            return self._fallback_sections()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_sections(self) -> List[Dict]:
        """
        Si aucune section identifiée : renvoyer les premières sections
        de niveau 1 (haut niveau) pour couvrir un maximum de contenu.
        """
        top_sections = [s for s in self.sections if s.get("level", 1) == 1]
        if not top_sections:
            top_sections = self.sections
        return [{**s, "score": 0.0} for s in top_sections[:self.max_sections]]

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def pages_for_query(self, query: str) -> List[int]:
        """
        Raccourci : retourne la liste plate des numéros de pages
        couverts par les sections ciblées.
        """
        sections = self.route(query)
        pages = set()
        for s in sections:
            for p in range(s["start_page"], s["end_page"] + 1):
                pages.add(p)
        return sorted(pages)

    def print_routing(self, query: str):
        """Affiche le résultat du routage de façon lisible."""
        sections = self.route(query)
        print(f"\n   🎯 Routage de : «{query}»")
        if sections:
            for s in sections:
                print(f"      → p.{s['start_page']}-{s['end_page']}  «{s['title']}»  (score {s.get('score', 0):.1f})")
        else:
            print("      → Aucune section ciblée (fallback : document entier)")
