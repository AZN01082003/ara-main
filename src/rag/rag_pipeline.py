"""
Pipeline RAG complet avec LLM (support Gemini)
Mode financier : prompt spécialisé + remplissage de tables prédéfinies
"""

import json
import re
import os
from typing import List, Dict, Optional
from src.retrieval.hybrid_retriever import HybridRetriever

# Imports pour différents LLM providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class RAGPipeline:
    """
    Pipeline RAG complet : Retrieval + Generation
    Support pour Gemini, Hugging Face, Ollama, OpenAI, et Groq
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_provider: str = "gemini",
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Args:
            retriever: Instance de HybridRetriever
            llm_provider: "gemini", "huggingface", "ollama", "openai", ou "groq"
            model_name: Nom du modèle à utiliser
                - Gemini: "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"
                - HuggingFace: "mistralai/Mistral-7B-Instruct-v0.2"
            api_key: Clé API (pour Gemini, Hugging Face, OpenAI ou Groq)
            temperature: Créativité du modèle (0 = déterministe, 1 = créatif)
            max_tokens: Longueur maximale de la réponse
        """
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configuration du LLM
        self._setup_llm(api_key)
        
        print(f"   ✅ RAG Pipeline configuré avec {llm_provider} ({model_name})")
    
    def _setup_llm(self, api_key: Optional[str]):
        """Configure le LLM selon le provider"""
        
        if self.llm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI non installé. Installez avec: pip install google-generativeai")
            
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key requise. Définissez GEMINI_API_KEY dans .env ou passez-la en paramètre")
            
            # Configurer Gemini
            genai.configure(api_key=api_key)
            
            # Créer le modèle avec configuration
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.95,
            }
            
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            print(f"   🔗 Connecté à Google Gemini API")
        
        elif self.llm_provider == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("Hugging Face non installé. Installez avec: pip install huggingface_hub")
            
            api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("Hugging Face API key requise. Définissez HUGGINGFACE_API_KEY dans .env")
            
            self.client = InferenceClient(token=api_key)
            print(f"   🔗 Connecté à Hugging Face API")
        
        elif self.llm_provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama non installé. Installez avec: pip install ollama")
            self.client = None
        
        elif self.llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI non installé. Installez avec: pip install openai")
            
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key requise. Définissez OPENAI_API_KEY dans .env")
            
            self.client = OpenAI(api_key=api_key)
        
        elif self.llm_provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq non installé. Installez avec: pip install groq")
            
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API key requise. Définissez GROQ_API_KEY dans .env")
            
            self.client = Groq(api_key=api_key)
        
        else:
            raise ValueError(f"Provider non supporté: {self.llm_provider}")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        return_sources: bool = True,
        mode: str = "standard",
    ) -> Dict:
        """
        Effectue une requête RAG complète.

        Args:
            question: Question de l'utilisateur
            top_k: Nombre de chunks à récupérer
            return_sources: Retourner les sources utilisées
            mode: "standard" (articles scientifiques) ou "financial" (rapports financiers)

        Returns:
            Dictionnaire avec 'answer', 'sources', 'metadata'
        """

        # --- Étape 1 : Retrieval ---
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        # --- Étape 2 : Contexte ---
        context = self._build_context(retrieved_docs)

        # --- Étape 3 : Prompt (selon le mode) ---
        if mode == "financial":
            prompt = self._build_financial_prompt(question, context)
        else:
            prompt = self._build_prompt(question, context)

        # --- Étape 4 : Génération ---
        answer = self._generate_answer(prompt)

        result = {
            'question': question,
            'answer': answer,
            'metadata': {
                'num_sources': len(retrieved_docs),
                'llm_provider': self.llm_provider,
                'model': self.model_name,
                'mode': mode,
            }
        }

        if return_sources:
            result['sources'] = self._format_sources(retrieved_docs)

        return result
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Construit le contexte à partir des documents récupérés"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            chunk_text = doc['text']
            score = doc['final_score']
            
            context_parts.append(
                f"[Document {i}] (Score: {score:.3f})\n{chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Construit le prompt complet pour le LLM"""
        
        # Prompt optimisé pour tous les modèles
        system_message = """You are a helpful research assistant analyzing scientific papers.
Your task is to answer questions based ONLY on the provided context from the paper.

Guidelines:
- Answer in a clear, concise manner
- Use information ONLY from the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific parts of the context when relevant
- Be precise and factual
- Avoid speculation or adding external knowledge"""

        prompt = f"""{system_message}

CONTEXT FROM THE PAPER:
{context}

QUESTION:
{question}

ANSWER:"""
        
        return prompt
    
    def _build_financial_prompt(self, question: str, context: str) -> str:
        """
        Prompt spécialisé pour les rapports financiers.
        Instruit le LLM à lire les tableaux Markdown, les valeurs
        numériques et les symboles financiers.
        """
        system_message = (
            "Tu es un analyste financier expert. Tu analyses des rapports financiers "
            "(bilans, comptes de résultat, flux de trésorerie, notes annexes).\n"
            "Tu dois répondre aux questions en te basant UNIQUEMENT sur le contexte fourni.\n\n"
            "Directives :\n"
            "- Les tableaux financiers sont en format Markdown (| col1 | col2 | …)\n"
            "- Lis attentivement les valeurs numériques, les variations (%), les unités "
            "(€, M€, k€, $)\n"
            "- Cite les chiffres précis du rapport quand c'est possible\n"
            "- Si une information est absente du contexte, dis-le clairement\n"
            "- Présente les données de manière structurée (listes, tableaux) quand pertinent\n"
            "- Tiens compte du contexte narratif ET des tableaux chiffrés\n"
            "- Ne spécule pas, ne complète pas avec des connaissances externes"
        )

        return (
            f"{system_message}\n\n"
            f"CONTEXTE DU RAPPORT FINANCIER :\n{context}\n\n"
            f"QUESTION :\n{question}\n\n"
            f"RÉPONSE :"
        )

    # ------------------------------------------------------------------
    # Remplissage de tables prédéfinies (cas d'usage principal)
    # ------------------------------------------------------------------

    def fill_predefined_table(
        self,
        table_template: Dict,
        top_k: int = 5,
    ) -> Dict:
        """
        Remplit une table prédéfinie à partir du rapport financier via RAG.

        Le LLM est appelé une fois par ligne (indicateur) pour extraire les
        valeurs correspondant à chaque colonne (périodes, exercices…).

        Args:
            table_template: {
                "table_name": str,          # ex. "Compte de résultat simplifié"
                "columns": [str, ...],       # ex. ["Indicateur", "2023", "2022", "Variation"]
                "rows": [str, ...]           # ex. ["Chiffre d'affaires", "EBITDA", "Résultat net"]
            }
            top_k: Chunks RAG récupérés par requête.

        Returns:
            {
                "table_name": str,
                "columns": [str, ...],
                "data": { row_name: { col: value, … }, … }
            }
        """
        table_name = table_template.get("table_name", "Tableau financier")
        columns = table_template.get("columns", [])
        rows = table_template.get("rows", [])

        if not columns or not rows:
            raise ValueError("table_template doit contenir 'columns' et 'rows'.")

        # Colonnes de données = toutes sauf la première ("Indicateur")
        data_columns = columns[1:] if len(columns) > 1 else columns

        filled_table = {
            "table_name": table_name,
            "columns": columns,
            "data": {},
        }

        print(f"\n   📋 Remplissage de la table : «{table_name}»")
        print(f"      {len(rows)} indicateurs × {len(data_columns)} colonnes\n")

        for row_name in rows:
            # Requête RAG ciblée sur l'indicateur
            query = f"{row_name} {table_name}"
            retrieved = self.retriever.retrieve(query, top_k=top_k)
            context = self._build_context(retrieved)

            prompt = self._build_table_fill_prompt(
                row_indicator=row_name,
                data_columns=data_columns,
                context=context,
                table_name=table_name,
            )

            raw_answer = self._generate_answer(prompt)
            parsed = self._parse_table_fill_response(raw_answer, data_columns)
            filled_table["data"][row_name] = parsed

            print(f"      ✓ {row_name} : {parsed}")

        return filled_table

    def _build_table_fill_prompt(
        self,
        row_indicator: str,
        data_columns: List[str],
        context: str,
        table_name: str,
    ) -> str:
        """Prompt d'extraction de valeurs pour une ligne de table."""
        cols_str = ", ".join(f'"{c}"' for c in data_columns)
        json_template = ", ".join(f'"{c}": "valeur ou N/A"' for c in data_columns)

        return (
            f"Tu es un analyste financier. Extrais les valeurs pour l'indicateur "
            f'"{row_indicator}" depuis le contexte du rapport financier ci-dessous.\n\n'
            f"CONTEXTE :\n{context}\n\n"
            f'TÂCHE : Extrais les valeurs de "{row_indicator}" dans la table '
            f'"{table_name}" pour les colonnes : {cols_str}.\n\n'
            f"Réponds UNIQUEMENT en JSON valide, sans texte supplémentaire :\n"
            f"{{{json_template}}}\n\n"
            f"Si une valeur n'est pas trouvée, mets \"N/A\".\n\nJSON :"
        )

    def _parse_table_fill_response(self, raw_response: str, data_columns: List[str]) -> Dict:
        """
        Parse la réponse JSON du LLM.
        Fallback sur N/A si le JSON est invalide ou incomplet.
        """
        json_match = re.search(r'\{[^{}]+\}', raw_response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # S'assurer que toutes les colonnes sont présentes
                return {col: parsed.get(col, "N/A") for col in data_columns}
            except json.JSONDecodeError:
                pass
        # Fallback
        return {col: "N/A" for col in data_columns}

    def format_filled_table_as_markdown(self, filled_table: Dict) -> str:
        """
        Convertit un tableau rempli (issu de fill_predefined_table) en Markdown.

        Returns:
            Chaîne Markdown prête à afficher ou à sauvegarder.
        """
        table_name = filled_table.get("table_name", "Tableau")
        columns = filled_table.get("columns", [])
        data = filled_table.get("data", {})

        lines = [f"## {table_name}\n"]

        # En-tête
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")

        # Données
        for row_name, values in data.items():
            row_cells = [row_name] + [str(values.get(col, "N/A")) for col in columns[1:]]
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)

    def _generate_answer(self, prompt: str) -> str:
        """Génère la réponse avec le LLM"""
        
        if self.llm_provider == "gemini":
            return self._generate_with_gemini(prompt)
        elif self.llm_provider == "huggingface":
            return self._generate_with_huggingface(prompt)
        elif self.llm_provider == "ollama":
            return self._generate_with_ollama(prompt)
        elif self.llm_provider == "openai":
            return self._generate_with_openai(prompt)
        elif self.llm_provider == "groq":
            return self._generate_with_groq(prompt)
    
    def _generate_with_gemini(self, prompt: str) -> str:
        """Génération avec Google Gemini"""
        try:
            print("   🔄 Génération en cours avec Gemini...")
            
            # Générer la réponse
            response = self.client.generate_content(prompt)
            
            # Vérifier si la réponse a été bloquée
            if not response.text:
                if hasattr(response, 'prompt_feedback'):
                    return f"⚠️ Réponse bloquée par les filtres de sécurité: {response.prompt_feedback}"
                return "⚠️ Aucune réponse générée. Le contenu a peut-être été filtré."
            
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            
            # Messages d'erreur plus clairs
            if "quota" in error_msg.lower() or "429" in error_msg:
                return "⚠️ Erreur: Quota API dépassé. Veuillez réessayer plus tard."
            elif "api key" in error_msg.lower() or "401" in error_msg:
                return "⚠️ Erreur: API key invalide. Vérifiez votre clé Gemini."
            elif "503" in error_msg or "unavailable" in error_msg.lower():
                return "⚠️ Erreur: Service temporairement indisponible. Réessayez dans quelques instants."
            else:
                return f"⚠️ Erreur Gemini: {error_msg}"
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        """Génération avec Hugging Face Inference API"""
        try:
            print("   🔄 Génération en cours avec Hugging Face...")
            
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            return response.strip()
            
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "rate limit" in error_msg.lower():
                return "⚠️ Erreur: Limite de requêtes atteinte. Veuillez réessayer dans quelques secondes."
            elif "503" in error_msg or "loading" in error_msg.lower():
                return "⚠️ Erreur: Le modèle est en cours de chargement. Veuillez réessayer dans 1-2 minutes."
            elif "unauthorized" in error_msg.lower() or "401" in error_msg:
                return "⚠️ Erreur: API key invalide. Vérifiez votre clé Hugging Face."
            else:
                return f"⚠️ Erreur Hugging Face: {error_msg}"
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Génération avec Ollama"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Erreur Ollama: {str(e)}. Vérifiez qu'Ollama est lancé et que le modèle '{self.model_name}' est installé."
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Génération avec OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur OpenAI: {str(e)}"
    
    def _generate_with_groq(self, prompt: str) -> str:
        """Génération avec Groq"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur Groq: {str(e)}"
    
    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """Formate les sources pour un affichage propre"""
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            sources.append({
                'source_number': i,
                'chunk_id': doc['chunk_id'],
                'text_preview': doc['text'][:200] + "...",
                'final_score': doc['final_score'],
                'chroma_score': doc.get('chroma_score', 0),
                'bm25_score': doc.get('bm25_score', 0)
            })
        
        return sources
    
    def chat(self, question: str, top_k: int = 5) -> str:
        """
        Version simple qui retourne juste la réponse
        (utile pour une interface de chat)
        """
        result = self.query(question, top_k=top_k, return_sources=False)
        return result['answer']
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur le pipeline RAG"""
        retriever_stats = self.retriever.get_statistics()
        
        return {
            'llm_provider': self.llm_provider,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'retriever': retriever_stats
        }