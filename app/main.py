from fastapi import FastAPI, HTTPException

from app.database import SessionManager
from app.models import HoneypotRequest
from app.voice_router import router as voice_router
from app.workflow.graph import run_honeypot_workflow


db_manager = SessionManager()

app = FastAPI(
    title="ScamBait AI - Honeypot Scam Detection",
    version="1.0.0",
    description="Active defense system that engages scammers and extracts forensic intelligence",
)

app.include_router(voice_router, prefix="/voice", tags=["voice"])


@app.post("/api/v1/honeypot")
async def honeypot_endpoint(request: HoneypotRequest):
    """Run the LangGraph workflow for an incoming honeypot message."""
    try:
        response = await run_honeypot_workflow(request)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Honeypot workflow failed: {e}") from e


@app.get("/")
async def root():
    return {"status": "online"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/stats")
async def get_stats():
    """Return aggregated stats for the dashboard."""
    return db_manager.get_stats()
