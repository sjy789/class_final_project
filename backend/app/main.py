from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import analyze_auto, evaluate_manual


app = FastAPI(title="Object Placement Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "object-placement-assistant"}


@app.post("/api/recommend")
async def recommend(
    background: UploadFile = File(...),
    foreground: UploadFile = File(...),
    top_k: int = Form(3),
    scale: float = Form(0.22),
    harmonize: bool = Form(True),
    explain: bool = Form(True),
) -> dict:
    try:
        return analyze_auto(
            await background.read(),
            await foreground.read(),
            top_k=top_k,
            scale=scale,
            harmonize=harmonize,
            explain=explain,
        )
    except Exception as exc:  # pragma: no cover - returned to UI for debugging.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluate")
async def evaluate(
    background: UploadFile = File(...),
    foreground: UploadFile = File(...),
    x: float = Form(0.35),
    y: float = Form(0.55),
    scale: float = Form(0.22),
    harmonize: bool = Form(True),
    explain: bool = Form(False),
) -> dict:
    try:
        return evaluate_manual(
            await background.read(),
            await foreground.read(),
            x_norm=x,
            y_norm=y,
            scale=scale,
            harmonize=harmonize,
            explain=explain,
        )
    except Exception as exc:  # pragma: no cover - returned to UI for debugging.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

