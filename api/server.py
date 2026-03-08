"""
API FastAPI — Financial RAG Evaluator

Endpoints :
  POST /api/upload                      Upload PDF + init pipeline ciblé
  POST /api/{pdf_id}/query              Question libre
  POST /api/{pdf_id}/table              Remplissage de table prédéfinie
  POST /api/{pdf_id}/evaluate           Lance l'évaluation (tâche asynchrone)
  GET  /api/{pdf_id}/tasks/{task_id}    Statut et résultats d'évaluation
  GET  /api/{pdf_id}/stats              Statistiques du cache d'extraction
  DELETE /api/{pdf_id}                  Supprime l'instance + fichiers temp

Démarrage :
  cd ara-main
  uvicorn api.server:app --reload --port 8000
"""

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_pipeline import RAGPipeline
from src.rag.targeted_pipeline import TargetedFinancialPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDAEAu5gWlmPRVnIc4i5eM2AEZQMph-I3o")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Suite d'évaluation
# ---------------------------------------------------------------------------
EVALUATION_SUITE: Dict[str, Dict] = {
    "Extraction numérique": {
        "weight": 1.5,
        "icon": "🔢",
        "description": "Capacité à extraire des chiffres précis",
        "questions": [
            "Quel est le chiffre d'affaires total mentionné dans le rapport ? Donne le montant exact avec l'unité.",
            "Quelle est la valeur exacte du résultat net ? Précise l'exercice concerné.",
            "Quel est le montant de la trésorerie ou des liquidités disponibles ?",
        ],
    },
    "Tableaux financiers": {
        "weight": 3.0,
        "icon": "📊",
        "description": "Extraction et structuration des tableaux",
        "questions": [
            "Extrait le compte de résultat complet sous forme de tableau Markdown avec les colonnes Indicateur, N, N-1 et Variation(%).",
            "Dresse le bilan condensé (actif et passif) avec les montants pour chaque poste principal.",
            "Présente les flux de trésorerie par catégorie (opérationnel, investissement, financement) avec les montants.",
        ],
    },
    "Indicateurs financiers": {
        "weight": 2.0,
        "icon": "📈",
        "description": "Ratios et KPIs financiers",
        "questions": [
            "Quel est l'EBITDA et comment a-t-il évolué par rapport à l'exercice précédent ?",
            "Quelle est la marge nette (%) et la marge opérationnelle (%) du dernier exercice ?",
            "Quel est le niveau d'endettement net et le ratio dette nette / EBITDA ?",
        ],
    },
    "Contexte et analyse": {
        "weight": 1.0,
        "icon": "🔍",
        "description": "Compréhension du contexte financier",
        "questions": [
            "Quels sont les trois principaux risques financiers identifiés dans le rapport ?",
            "Comment la direction commente-t-elle les résultats de l'exercice et quels objectifs sont fixés ?",
            "Quelles sont les perspectives financières (guidance) annoncées pour l'exercice suivant ?",
        ],
    },
}

# ---------------------------------------------------------------------------
# Etat en mémoire
# ---------------------------------------------------------------------------
pipelines: Dict[str, Dict] = {}   # pdf_id → {pipeline, filename, tmp_dir, …}
tasks: Dict[str, Dict] = {}        # task_id → {status, progress, results, error}
executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Financial RAG Evaluator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    mode: str = "financial"
    top_k: int = 5


class TableRequest(BaseModel):
    table_template: Dict[str, Any]


class EvaluateRequest(BaseModel):
    category: Optional[str] = None  # None → toutes les catégories


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend non trouvé — placez index.html dans api/static/</h1>")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload un PDF financier et initialise le pipeline ciblé.
    Seul le scan structurel est effectué ici (~50 ms) ; l'extraction complète
    est déclenchée de manière lazy à chaque requête.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    pdf_id = uuid.uuid4().hex[:10]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"finrag_{pdf_id}_"))
    pdf_path = tmp_dir / file.filename

    # Sauvegarde du fichier
    content = await file.read()
    pdf_path.write_bytes(content)

    # Initialisation pipeline dans le thread pool (CPU-bound)
    import asyncio
    loop = asyncio.get_event_loop()

    def _init():
        rag = RAGPipeline(
            retriever=None,
            llm_provider="gemini",
            model_name=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            temperature=0.2,
            max_tokens=2048,
        )
        out_dir = tmp_dir / "outputs"
        out_dir.mkdir(exist_ok=True)
        return TargetedFinancialPipeline(
            pdf_path=str(pdf_path),
            rag_pipeline=rag,
            embedding_model=EMBEDDING_MODEL,
            chunk_size=800,
            overlap=100,
            language="fr",
            top_k=5,
            output_dir=str(out_dir),
        )

    try:
        pipeline = await loop.run_in_executor(executor, _init)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Erreur d'initialisation : {e}")

    pipelines[pdf_id] = {
        "pipeline": pipeline,
        "filename": file.filename,
        "tmp_dir": str(tmp_dir),
        "created_at": time.time(),
    }

    smap = pipeline.structural_map
    return {
        "pdf_id": pdf_id,
        "filename": file.filename,
        "title": smap.get("title") or file.filename,
        "total_pages": smap.get("total_pages", 0),
        "sections": smap.get("sections", []),
    }


@app.post("/api/{pdf_id}/query")
async def query_pdf(pdf_id: str, req: QueryRequest):
    """Question libre sur le PDF avec extraction ciblée."""
    pipeline = _get_pipeline(pdf_id)

    import asyncio
    loop = asyncio.get_event_loop()

    def _run():
        return pipeline.query(req.question, mode=req.mode)

    try:
        result = await loop.run_in_executor(executor, _run)
        result["cache_stats"] = pipeline.get_cache_stats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/{pdf_id}/table")
async def fill_table(pdf_id: str, req: TableRequest):
    """Remplit une table prédéfinie avec extraction ciblée."""
    pipeline = _get_pipeline(pdf_id)

    import asyncio
    loop = asyncio.get_event_loop()

    def _run():
        filled = pipeline.fill_predefined_table(req.table_template)
        md = pipeline.rag.format_filled_table_as_markdown(filled)
        return {"filled": filled, "markdown": md}

    try:
        result = await loop.run_in_executor(executor, _run)
        result["cache_stats"] = pipeline.get_cache_stats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/{pdf_id}/evaluate")
async def start_evaluation(
    pdf_id: str, req: EvaluateRequest, background_tasks: BackgroundTasks
):
    """
    Lance l'évaluation en tâche de fond.
    Retourne immédiatement un task_id à surveiller via GET /tasks/{task_id}.
    """
    _get_pipeline(pdf_id)  # vérification existence

    task_id = uuid.uuid4().hex[:8]
    total = sum(
        len(v["questions"])
        for k, v in EVALUATION_SUITE.items()
        if req.category is None or k == req.category
    )
    tasks[task_id] = {
        "status": "running",
        "progress": {"current": 0, "total": total},
        "results": None,
        "error": None,
    }
    background_tasks.add_task(_run_evaluation, pdf_id, task_id, req.category)
    return {"task_id": task_id, "total_questions": total}


@app.get("/api/{pdf_id}/tasks/{task_id}")
async def get_task(pdf_id: str, task_id: str):
    """Retourne le statut et les résultats d'une tâche d'évaluation."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tâche introuvable.")
    return tasks[task_id]


@app.get("/api/{pdf_id}/stats")
async def get_stats(pdf_id: str):
    return _get_pipeline(pdf_id).get_cache_stats()


@app.delete("/api/{pdf_id}")
async def delete_pdf(pdf_id: str):
    if pdf_id not in pipelines:
        raise HTTPException(status_code=404, detail="PDF introuvable.")
    data = pipelines.pop(pdf_id)
    shutil.rmtree(data["tmp_dir"], ignore_errors=True)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pipeline(pdf_id: str) -> TargetedFinancialPipeline:
    if pdf_id not in pipelines:
        raise HTTPException(status_code=404, detail="PDF introuvable. Uploadez-le d'abord.")
    return pipelines[pdf_id]["pipeline"]


def _score_answer(rag: RAGPipeline, question: str, answer: str, category: str) -> Dict:
    """
    Utilise le LLM pour évaluer la qualité d'une réponse RAG financière.
    Retourne {"score": float 0-10, "comment": str}.
    """
    prompt = (
        f"Tu es un expert en analyse de rapports financiers. "
        f"Évalue la qualité de cette réponse produite par un système RAG.\n\n"
        f"Catégorie : {category}\n"
        f"Question : {question}\n\n"
        f"Réponse du système :\n{answer[:2000]}\n\n"
        f"Critères (chacun sur 10) :\n"
        f"- Précision des données chiffrées extraites\n"
        f"- Qualité structurelle (tableaux bien formés si applicable)\n"
        f"- Pertinence et exhaustivité financière\n"
        f"- Terminologie financière correcte\n\n"
        f"Réponds UNIQUEMENT avec ce JSON, sans texte avant ni après :\n"
        f'{{ "score": <moyenne 0-10>, "comment": "<justification en 1 phrase>" }}'
    )
    try:
        raw = rag._generate_answer(prompt).strip()
        # Nettoyer les balises markdown éventuelles
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        data["score"] = round(max(0.0, min(10.0, float(data.get("score", 5)))), 1)
        return data
    except Exception:
        return {"score": 5.0, "comment": "Score non disponible (erreur de parsing)"}


def _run_evaluation(pdf_id: str, task_id: str, category_filter: Optional[str]):
    """Tâche de fond : exécute la suite d'évaluation et met à jour tasks[task_id]."""
    try:
        pipeline = pipelines[pdf_id]["pipeline"]
        rag = pipeline.rag

        suite = {
            k: v for k, v in EVALUATION_SUITE.items()
            if category_filter is None or k == category_filter
        }

        cat_results: Dict[str, Dict] = {}
        current = 0

        for cat_name, cat_data in suite.items():
            q_results = []
            for question in cat_data["questions"]:
                t0 = time.time()
                try:
                    resp = pipeline.query(question, mode="financial")
                    answer = resp.get("answer", "")
                    targeted_pages = resp.get("targeted_pages", [])

                    score_data = _score_answer(rag, question, answer, cat_name)
                except Exception as e:
                    answer = f"Erreur : {e}"
                    targeted_pages = []
                    score_data = {"score": 0.0, "comment": "Erreur d'exécution"}

                q_results.append({
                    "question": question,
                    "answer": answer,
                    "score": score_data["score"],
                    "comment": score_data["comment"],
                    "targeted_pages": targeted_pages,
                    "latency_ms": int((time.time() - t0) * 1000),
                })

                current += 1
                tasks[task_id]["progress"]["current"] = current

            scores = [r["score"] for r in q_results]
            avg = round(sum(scores) / len(scores), 1) if scores else 0.0
            cat_results[cat_name] = {
                "icon": cat_data["icon"],
                "description": cat_data["description"],
                "questions": q_results,
                "average_score": avg,
                "weight": cat_data["weight"],
            }

        # Score global pondéré
        total_weight = sum(v["weight"] for v in cat_results.values())
        weighted = (
            sum(v["average_score"] * v["weight"] for v in cat_results.values())
            / total_weight
            if total_weight > 0 else 0.0
        )
        overall = round(weighted, 1)

        tasks[task_id].update({
            "status": "done",
            "results": {
                "categories": cat_results,
                "overall_score": overall,
                "max_score": 10,
                "grade": _score_to_grade(overall),
                "cache_stats": pipeline.get_cache_stats(),
            },
        })
    except Exception as e:
        tasks[task_id].update({"status": "error", "error": str(e)})


def _score_to_grade(score: float) -> str:
    if score >= 9:   return "Excellent"
    if score >= 7.5: return "Très bien"
    if score >= 6:   return "Bien"
    if score >= 5:   return "Acceptable"
    if score >= 3:   return "Insuffisant"
    return "Très insuffisant"


# ---------------------------------------------------------------------------
# Entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
