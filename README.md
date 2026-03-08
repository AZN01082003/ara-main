# FinRAG Evaluator

Interface web pour **uploader un rapport financier PDF** et évaluer la qualité de compréhension du pipeline RAG : extraction de chiffres, tableaux financiers, indicateurs et contexte.

---

## Prérequis

| Outil | Version minimale |
|---|---|
| Python | **3.11** recommandé (3.10 fonctionne, 3.8 non supporté) |
| VS Code | dernière version |
| Tesseract OCR | 4.x ([installer](https://github.com/tesseract-ocr/tesseract#installing-tesseract)) |
| Clé API Gemini | [console Google AI Studio](https://aistudio.google.com/app/apikey) |

> **Windows** : installez aussi [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) et ajoutez `bin/` au `PATH` (requis par `pdf2image`).

---

## Installation rapide

### 1 — Cloner et ouvrir dans VS Code

```bash
git clone <url-du-repo>
cd ara-main
code .
```

### 2 — Créer et activer un environnement virtuel

```bash
# Créer le venv (une seule fois)
python -m venv .venv

# Activer — Linux / macOS
source .venv/bin/activate

# Activer — Windows PowerShell
.venv\Scripts\Activate.ps1
```

> VS Code détecte `.venv` automatiquement. Si ce n'est pas le cas :  
> `Ctrl+Shift+P` → **Python: Select Interpreter** → choisir `.venv`

### 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4 — Télécharger le modèle spaCy

```bash
python -m spacy download fr_core_news_sm
```

### 5 — Configurer la clé API Gemini

Créez un fichier `.env` à la racine du projet :

```dotenv
GEMINI_API_KEY=AIza...votre_clé...
```

> Le serveur lit `GEMINI_API_KEY` via `os.getenv()`. Ne commitez jamais ce fichier.

---

## Lancer le serveur

### Option A — Terminal VS Code intégré

```bash
# Windows — sans --reload (obligatoire, voir note ci-dessous)
uvicorn api.server:app --port 8000

# Linux / macOS — --reload optionnel
uvicorn api.server:app --reload --port 8000
```

Ouvrez ensuite **http://localhost:8000** dans votre navigateur.

> **Windows — pourquoi pas `--reload` ?**
> Sur Windows, `--reload` spawne un sous-process via `multiprocessing.spawn`.
> Ce spawn réimporte toute l'app, y compris les extensions PyO3 de `cryptography`,
> ce qui déclenche : *"PyO3 modules compiled for CPython 3.8 or older may only be initialized once"*.
> Sans `--reload`, aucun sous-process n'est créé → l'erreur disparaît.
> Pour recharger l'app après un changement de code : **Ctrl+C** puis relancer.

### Option B — Configuration de lancement VS Code

Créez `.vscode/launch.json` :

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FinRAG — Serveur API",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["api.server:app", "--reload", "--port", "8000"],
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal"
    }
  ]
}
```

Appuyez sur **F5** pour démarrer.

---

## Utiliser l'interface

### Onglet Upload
Glissez-déposez un rapport financier PDF (bilan annuel, rapport semestriel, etc.).  
Le plan du document (sections + plages de pages) s'affiche immédiatement.

### Onglet Interrogation
Posez des questions en langage naturel. Le système extrait uniquement les pages pertinentes (**lazy extraction**) et affiche :
- La réponse générée (Markdown rendu)
- Les pages ciblées + latence
- Les sources dépliables
- La progression du cache d'extraction

### Onglet Tableaux
Choisissez un modèle prédéfini ou construisez votre propre table :

| Modèle | Lignes extraites |
|---|---|
| Compte de résultat | CA, EBITDA, EBIT, Résultat net, Marge |
| Bilan condensé | Actif, Passif, Capitaux propres, Dettes |
| Flux de trésorerie | Opérationnel, Investissement, Financement |
| Ratios financiers | Marges, endettement, liquidité |

Export CSV disponible après extraction.

### Onglet Évaluation
Lance une **suite de 12 questions** réparties en 4 catégories pondérées.  
Chaque réponse est scorée **0–10 par le LLM**. Résultats en temps réel :
- Radar Chart par catégorie
- Score global pondéré + mention (Excellent → Très insuffisant)
- Détail accordéon par question (réponse, score, pages, latence)

| Catégorie | Poids | Ce qui est testé |
|---|---|---|
| Extraction numérique | ×1.5 | Précision des chiffres bruts |
| **Tableaux financiers** | **×3.0** | Structuration et complétude des tables |
| Indicateurs financiers | ×2.0 | EBITDA, marges, endettement |
| Contexte et analyse | ×1.0 | Risques, perspectives, commentaires |

---

## Structure du projet

```
ara-main/
├── api/
│   ├── server.py          # Backend FastAPI (endpoints + suite d'évaluation)
│   └── static/
│       └── index.html     # Frontend JS (4 onglets, Chart.js, Markdown)
├── src/
│   ├── extraction/        # PDFExtractor, PDFStructureScanner
│   ├── preprocessing/     # TextCleaner, Chunker (financier)
│   ├── embeddings/        # SentenceTransformer wrapper
│   ├── indexing/          # ChromaDB + BM25
│   ├── retrieval/         # HybridRetriever (sémantique + BM25)
│   ├── routing/           # SectionRouter (pages ciblées)
│   └── rag/
│       ├── rag_pipeline.py        # Pipeline principal (query, fill_table, score)
│       └── targeted_pipeline.py   # Pipeline ciblé (lazy extraction)
├── main_financial.py      # CLI pour tests rapides
├── requirements.txt
├── .env                   # ← à créer (non versionné)
└── README.md
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Requis** — clé API Google Gemini |
| `PORT` | `8000` | Port d'écoute (uvicorn) |

---

## Dépannage

| Erreur | Solution |
|---|---|
| `PyO3 modules compiled for CPython 3.8 or older` | `pip install "pdfplumber==0.7.6" "pdfminer.six==20211012"` — voir ci-dessous |
| `ModuleNotFoundError: fitz` | `pip install PyMuPDF` |
| `tesseract is not installed` | Installer Tesseract + ajouter au PATH |
| `poppler not found` | Installer Poppler (Windows) ou `apt install poppler-utils` |
| `fr_core_news_sm not found` | `python -m spacy download fr_core_news_sm` |
| Erreur 429 Gemini | Quota dépassé — attendre ou changer de clé |
| Port 8000 déjà utilisé | `uvicorn api.server:app --port 8001` |

### Erreur PyO3 — "initialized once per interpreter process"

```
ImportError: PyO3 modules compiled for CPython 3.8 or older
  may only be initialized once per interpreter process
```

**Cause réelle** : `pdfplumber ≥ 0.9` exige `pdfminer.six ≥ 20220524`, qui a
introduit une dépendance vers `cryptography`. `cryptography ≥ 38` utilise des
extensions Rust compilées avec PyO3 via le stable ABI `abi3-cp38`. Sur
**Python 3.10.0rc1** (release candidate), la détection de version de PyO3 présente
un bug et refuse de charger ces wheels.

**Fix — réinstaller les seuls paquets concernés** :

```powershell
pip install "pdfplumber==0.7.6" "pdfminer.six==20211012"
```

`pdfminer.six 20211012` (Oct 2021) est antérieur à l'ajout de `cryptography` comme
dépendance. Toutes les APIs utilisées (`extract_text`, `extract_tables`) sont
disponibles dans cette version.

Puis relancer :
```powershell
uvicorn api.server:app --port 8000
```

> **Solution définitive** : installer [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
> (ou 3.11.x), recréer le venv et relancer `pip install -r requirements.txt`.
> Les versions stables `3.10.x` n'ont pas ce bug.
