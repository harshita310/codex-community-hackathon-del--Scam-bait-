from app.database import SessionManager

db_manager = SessionManager()
from app.models import HoneypotRequest, JudgeResponse, ResponseMeta

@app.post("/api/v1/honeypot")
async def honeypot_endpoint(request: HoneypotRequest):
    """Main honeypot endpoint stub."""
    # Load or initialize session
    session_state = db_manager.get_session(request.sessionId) or {}
    print(f"Loaded session for {request.sessionId}")


    try:
        # Placeholder for complex agent workflow
        from app.workflow.graph import run_honeypot_workflow
        session_state = run_honeypot_workflow(request.message.text, session_state)
        reply = "Let me get my glasses..."
        status = "success"
    except Exception as e:
        reply = "I'm sorry, my phone is acting up."
        status = "error"
        db_manager.save_session(request.sessionId, session_state)
    return {"status": status, "reply": reply, "extracted": session_state.get("extractedIntelligence", {})}

from fastapi import FastAPI

app = FastAPI(
    title="ScamBait AI - Honeypot Scam Detection",
    version="1.0.0",
    description="Active defense system that engages scammers and extracts forensic intelligence"
)

@app.get("/")
async def root():
    return {"status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/stats")
async def get_stats():
    """Returns aggregated stats for the dashboard."""
    # Directly query production database for live stats
    return db_manager.get_stats()
