import base64

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import OPENAI_API_KEY, OPENAI_VISION_MODEL
from app.utils import logger


_OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class VisionAnalysis(BaseModel):
    is_scam_image: bool = False
    indicators_found: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    extracted_entities: dict = Field(default_factory=dict)


async def analyze_scam_image(image_url: str = None, image_base64: str = None) -> dict:
    """
    Analyze an image for scam indicators using OpenAI vision.
    """
    if _OPENAI_CLIENT is None:
        logger.warning("Vision analysis requested but OPENAI_API_KEY is missing.")
        return VisionAnalysis().model_dump()

    if not image_url and not image_base64:
        return VisionAnalysis().model_dump()

    try:
        image_reference = image_url
        if image_base64 and not image_url:
            encoded_payload = image_base64
            try:
                base64.b64decode(image_base64, validate=True)
            except Exception:
                encoded_payload = base64.b64encode(image_base64.encode("utf-8")).decode("utf-8")
            image_reference = f"data:image/png;base64,{encoded_payload}"

        response = await _OPENAI_CLIENT.beta.chat.completions.parse(
            model=OPENAI_VISION_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze this image for scam indicators. Check for: fake QR payment codes, "
                        "spoofed bank/government logos, threatening legal notices, APK install requests, "
                        "cryptocurrency wallet addresses, fake UPI handles. Return the structured result only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image for scam indicators. Check for: fake QR payment codes, "
                                "spoofed bank/government logos, threatening legal notices, APK install requests, "
                                "cryptocurrency wallet addresses, fake UPI handles. Return JSON with: "
                                "is_scam_image, indicators_found, confidence, extracted_entities."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_reference},
                        },
                    ],
                },
            ],
            response_format=VisionAnalysis,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return VisionAnalysis().model_dump()
        return parsed.model_dump()
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}", exc_info=True)
        return VisionAnalysis().model_dump()


async def should_analyze_image(message_text: str) -> bool:
    text = (message_text or "").lower()
    signals = [
        "[image]",
        "image attached",
        "photo attached",
        "screenshot",
        ".jpg",
        ".jpeg",
        ".png",
        "qr code",
        "scan this",
    ]
    return any(signal in text for signal in signals)
