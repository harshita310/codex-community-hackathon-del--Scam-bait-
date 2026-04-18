from fastapi import APIRouter, Request, Response, WebSocket
from app.utils import logger
from app.services.audio_orchestrator import AudioOrchestrator

router = APIRouter()

@router.api_route("/incoming", methods=["GET", "POST"])
async def incoming_call(request: Request):
    """
    Handle incoming calls from Twilio.
    Returns TwiML to connect to the Media Stream.
    """
    if request.method == "GET":
        return Response(content="Twilio Webhook Ready. Please configure Twilio to use POST.", media_type="text/plain")

    host = request.headers.get("host")
    ws_url = f"wss://{host}/voice/stream"
    
    logger.info(f"Incoming call. Redirecting to Media Stream at {ws_url}")

    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>
"""
    return Response(content=twiml_response, media_type="application/xml")


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    """
    orchestrator = AudioOrchestrator(websocket)
    await orchestrator.start()
