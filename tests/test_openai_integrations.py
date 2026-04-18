import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from app.agents import detection, extraction, vision
from app.services import memory_service, realtime_service, voice_service


def test_detect_scam_with_openai(monkeypatch):
    payload = {
        "is_scam": True,
        "scam_type": "DIGITAL_ARREST",
        "confidence": 0.91,
        "extracted_entities": {
            "phone_numbers": ["9876543210"],
            "upi_ids": ["scammer@paytm"],
            "links": ["https://fraud.example"],
        },
    }

    class DummyCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=json.dumps(payload))
                                )
                            ]
                        )
                    )
                ]
            )

    monkeypatch.setattr(
        detection,
        "_OPENAI_CLIENT",
        SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions())),
    )

    result = asyncio.run(detection.detect_scam_with_openai("Transfer money to avoid arrest"))
    assert result["is_scam"] is True
    assert result["scam_type"] == "DIGITAL_ARREST"
    assert result["extracted_entities"]["upi_ids"] == ["scammer@paytm"]


def test_extract_with_openai_and_regex_merge(monkeypatch):
    parsed = extraction.ExtractedIntelligence(
        phone_numbers=["9876543210"],
        upi_ids=["9876@paytm"],
        bank_accounts=[],
        crypto_wallets=[],
        phishing_links=["https://fraud.example"],
        aadhaar_numbers=[],
        amounts=["90000000"],
    )

    class DummyCompletions:
        async def parse(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )

    monkeypatch.setattr(
        extraction,
        "_OPENAI_CLIENT",
        SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))),
    )

    result = asyncio.run(
        extraction.extract_intelligence(
            [{"sender": "scammer", "text": "send 9 8 7 6 at paytm and transfer nine crore rupees"}]
        )
    )
    assert "9876@paytm" in result["upiIds"]
    assert "90000000" in result["amounts"] or "nine crore" not in result["amounts"]


def test_transcribe_audio(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    class DummyTranscriptions:
        def create(self, **kwargs):
            return SimpleNamespace(text="namaste")

    monkeypatch.setattr(
        voice_service,
        "_OPENAI_CLIENT",
        SimpleNamespace(audio=SimpleNamespace(transcriptions=DummyTranscriptions())),
    )

    transcript = asyncio.run(voice_service.transcribe_audio(str(audio_path)))
    assert transcript == "namaste"


def test_synthesize_speech(monkeypatch, tmp_path):
    output_path = tmp_path / "speech.wav"

    class DummyStreamingResponse:
        def __init__(self, destination: Path):
            self.destination = destination

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream_to_file(self, file_path: str):
            Path(file_path).write_bytes(b"wav-bytes")

    class DummySpeech:
        def __init__(self):
            self.with_streaming_response = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            return DummyStreamingResponse(output_path)

    monkeypatch.setattr(
        voice_service,
        "_OPENAI_CLIENT",
        SimpleNamespace(audio=SimpleNamespace(speech=DummySpeech())),
    )

    generated_path = asyncio.run(
        voice_service.synthesize_speech("Haan beta, mujhe samajh nahi aaya", str(output_path))
    )
    assert Path(generated_path).exists()
    assert Path(generated_path).read_bytes() == b"wav-bytes"


def test_analyze_scam_image(monkeypatch):
    parsed = vision.VisionAnalysis(
        is_scam_image=True,
        indicators_found=["fake QR code", "spoofed logo"],
        confidence=0.88,
        extracted_entities={"upi_ids": ["fraud@paytm"]},
    )

    class DummyCompletions:
        async def parse(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )

    monkeypatch.setattr(
        vision,
        "_OPENAI_CLIENT",
        SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))),
    )

    result = asyncio.run(vision.analyze_scam_image(image_url="https://example.com/qr.png"))
    assert result["is_scam_image"] is True
    assert "fake QR code" in result["indicators_found"]


def test_memory_service_similarity(monkeypatch):
    monkeypatch.setattr(
        memory_service,
        "KNOWN_SCAM_PATTERNS",
        [
            {"name": "sim_block", "text": "Your SIM will be blocked"},
            {"name": "safe_chat", "text": "Let us have coffee tomorrow"},
        ],
    )
    monkeypatch.setattr(
        memory_service,
        "KNOWN_SCAM_PATTERN_EMBEDDINGS",
        {
            "sim_block": [1.0, 0.0],
            "safe_chat": [0.0, 1.0],
        },
    )
    monkeypatch.setattr(memory_service, "embed_text", lambda text: [0.9, 0.1])

    pattern, score = memory_service.find_closest_scam_pattern("Your SIM will be blocked today")
    assert pattern == "sim_block"
    assert score > 0.75
    assert memory_service.is_semantically_similar_to_scam("Your SIM will be blocked today")


def test_realtime_service_session_shape():
    session = realtime_service.build_realtime_session()
    assert session["model"]
    assert session["voice"]
