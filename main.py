import os
import base64
import tempfile
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Robot AI Cloud")


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "deepseek/deepseek-chat"
)


# ============================================================
# BASIC CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "robot-ai-cloud",
        "message": "Robot AI cloud backend is running"
    }


@app.get("/health")
def health():
    return {
        "server": True,
        "groq": bool(GROQ_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "elevenlabs": bool(ELEVENLABS_API_KEY),
        "voice_id": bool(ELEVENLABS_VOICE_ID)
    }


# ============================================================
# GROQ SPEECH TO TEXT
# ============================================================

def speech_to_text(audio_bytes: bytes) -> str:

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing")

    # Save temporary WAV file
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        url = "https://api.groq.com/openai/v1/audio/transcriptions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        with open(temp_path, "rb") as audio_file:

            files = {
                "file": (
                    "robot.wav",
                    audio_file,
                    "audio/wav"
                )
            }

            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "json",
                "temperature": "0"
            }

            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Groq error {response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        text = result.get("text", "").strip()

        return text

    finally:

        if temp_path:
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


# ============================================================
# OPENROUTER AI
# ============================================================

def ask_ai(user_text: str) -> str:

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """
You are the AI assistant inside a small friendly robot.

Your job is to have a natural conversation with the robot's owner.

Rules:
- Keep answers short and conversational.
- Normally answer in simple English for the first robot test.
- Do not write long essays.
- Do not use markdown unless necessary.
- Do not describe yourself as a cloud server.
- Sound friendly and natural.
"""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "temperature": 0.7,
        "max_tokens": 120
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    try:
        answer = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected OpenRouter response: {result}"
        )

    return answer.strip()


# ============================================================
# ELEVENLABS TEXT TO SPEECH
# ============================================================

def text_to_speech(text: str) -> bytes:

    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing")

    if not ELEVENLABS_VOICE_ID:
        raise RuntimeError("ELEVENLABS_VOICE_ID is missing")

    url = (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        f"{ELEVENLABS_VOICE_ID}"
    )

    params = {
        "output_format": "pcm_16000"
    }

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/pcm"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2"
    }

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=payload,
        timeout=90
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs error {response.status_code}: "
            f"{response.text}"
        )

    return response.content


# ============================================================
# COMPLETE ROBOT CONVERSATION
# ============================================================

@app.post("/talk")
async def talk(
    audio: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check audio
    # --------------------------------------------------------

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file"
        )

    # Protect server from accidentally huge uploads
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Audio file is too large"
        )

    try:

        # ----------------------------------------------------
        # 1. SPEECH → TEXT
        # ----------------------------------------------------

        user_text = speech_to_text(audio_bytes)

        if not user_text:
            return JSONResponse({
                "success": True,
                "heard": "",
                "reply": "",
                "audio_base64": ""
            })

        # ----------------------------------------------------
        # 2. TEXT → AI
        # ----------------------------------------------------

        ai_reply = ask_ai(user_text)

        # ----------------------------------------------------
        # 3. AI TEXT → SPEECH
        # ----------------------------------------------------

        audio_response = text_to_speech(ai_reply)

        # ----------------------------------------------------
        # 4. Convert audio to Base64
        # ----------------------------------------------------

        audio_base64 = base64.b64encode(
            audio_response
        ).decode("ascii")

        return JSONResponse({
            "success": True,
            "heard": user_text,
            "reply": ai_reply,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "audio_base64": audio_base64
        })

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
