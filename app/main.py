from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.database import SessionManager
from app.models import HoneypotRequest, Message
from app.voice_router import router as voice_router
from app.websockets import manager
from app.workflow.graph import run_honeypot_workflow


db_manager = SessionManager()

app = FastAPI(
    title="ScamBait AI - Honeypot Scam Detection",
    version="1.0.0",
    description="Active defense system that engages scammers and extracts forensic intelligence",
)

app.include_router(voice_router, prefix="/voice", tags=["voice"])


def _dashboard_message_event(
    *,
    session_id: str,
    message: Message | dict,
    webhook_id: str,
) -> dict:
    if isinstance(message, Message):
        payload = message.model_dump()
    else:
        payload = message

    timestamp = payload.get("timestamp") or datetime.utcnow().isoformat() + "Z"
    event_data = {
        "webhookId": webhook_id,
        "sessionId": session_id,
        "sender": payload.get("sender", "user"),
        "text": payload.get("text", ""),
        "timestamp": str(timestamp),
    }

    # Keep fields both at top-level and under data. The current dashboard reads
    # top-level fields, while the event schema declares data as the payload.
    return {"type": "new_message", "data": event_data, **event_data}


@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """Stream live honeypot traffic to the real-time dashboard."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/v1/honeypot")
async def honeypot_endpoint(request: HoneypotRequest):
    """Run the LangGraph workflow for an incoming honeypot message."""
    try:
        await manager.broadcast(
            _dashboard_message_event(
                session_id=request.sessionId,
                message=request.message,
                webhook_id=f"{request.sessionId}-in-{datetime.utcnow().timestamp()}",
            )
        )

        response = await run_honeypot_workflow(request)

        await manager.broadcast(
            _dashboard_message_event(
                session_id=request.sessionId,
                message={
                    "sender": "user",
                    "text": response.reply,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                webhook_id=f"{request.sessionId}-out-{datetime.utcnow().timestamp()}",
            )
        )
        await manager.broadcast({
            "type": "stats_update",
            "data": db_manager.get_stats(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

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
