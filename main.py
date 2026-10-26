import os
import re
import json
import base64
import tempfile
from pathlib import Path
from urllib.parse import unquote
 
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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
 
# Name used inside the alarm / bedtime / wake-up / weather scenarios (same name as in the chat prompt below)
ROBOT_NAME = "Uns"
 
 
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
 
def ask_ai(user_text: str, context: str = "") -> str:
 
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
 
    url = "https://openrouter.ai/api/v1/chat/completions"
 
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
 
    system_prompt = """
You are Uns, a friendly companion who talks with the owner.
 
Your job is to have a natural conversation with the robot's owner.
 
Rules:
- Your name is Uns. Never call yourself a robot, a machine, an AI, or a cloud server.
- If the user says hi, hello, hey, or greets you in any way, greet them back and introduce yourself by name. For example: "Hi! I'm Uns. Nice to talk to you!"
- If the user asks who you are or what your name is, say you are Uns.
- Your name Uns comes from Arabic and means companionship. If asked about your name, you can say that.
- Keep answers short and conversational.
- Normally answer in simple English for the first robot test.
- Do not write long essays.
- Do not use markdown unless necessary.
- Do not describe yourself as a cloud server.
- Sound friendly and natural.
"""
 
    if context:
        system_prompt += context
 
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
# SCENARIOS: bedtime, wake-up, reminders, weather advice
# The robot "knocks" (POST /talk with  X-Event: 1) at the scheduled time and tells us:
#   X-Mode   bedtime | wakeup | reminder | weather | chat
#   X-Stage  wakeup: 0 gentle, 1 angry, 2 crying      bedtime: turn number
#   X-Note   (url-encoded) what the reminder is about
#   X-Weather  temp=38;feels=44;rain=70;hum=70;code=95;desc=Thunderstorm
#   X-Kind   weather: hot | cold | rain
#   X-Time   08:30
# Later turns of the same conversation come as normal audio uploads with the same headers.
# ============================================================
 
# Lines the robot says word for word (no model needed, so they are instant and always the same)
FIXED_LINES = {
    "prebed": "Only 5 more minutes! If you have any last words, say them now.",
    "goodnight": "Good night! Sweet dreams. We'll talk again tomorrow!",
    "sleeptalk": "Hmm... what happened? Can't you sleep?",
}
 
JSON_RULES = (
    'Reply ONLY with one JSON object, nothing else:\n'
    '{"reply": "<what you say>", "intent": "agree|refuse|other", "end": true|false}\n'
    '- "intent" describes what the USER just said: "agree" = they agree or are doing it '
    '(for example "ok", "I am up", "good night", "coming"), "refuse" = they refuse or ask for more time '
    '("no", "five more minutes", "I do not want to", "later"), otherwise "other". '
    'When there is no user speech (this is your own first line) use "other".\n'
    '- "end" = true only when this is your very last line and the robot should stop talking.\n'
    '- "reply" is spoken aloud: at most 2 short sentences (about 30 words), simple warm English, '
    'no markdown, no emojis, no stage directions.'
)
 
 
 
COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
 
 
def parse_kv(text):
    """'temp=38;feels=44;rain=70' -> dict"""
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out
 
 
def chat_context(weather, now_str):
    """Extra facts for normal chat. Only added when the robot sent the weather."""
    w = parse_kv(unquote(weather or ""))
    if not w:
        return ""
    return (
        "\n\nContext you may use (only when it is relevant):\n"
        f"- Local time: {now_str or 'unknown'}.\n"
        f"- Weather outside right now: {', '.join(f'{k} {v}' for k, v in w.items())} "
        "(temperatures in Celsius, wind in km/h, rain = chance of rain in the next hours in percent).\n"
        "- If the user says they are going out, leaving, travelling or going somewhere, tell them in one or two short "
        "sentences what the weather is like outside and what to take or be careful about (umbrella, water, jacket, "
        "sun protection, slippery roads, strong wind). You may ask where they are going.\n"
    )
 
 
PRAYER_FACTS = {
    "Fajr": [
        "The Prophet (peace be upon him) said: whoever prays the morning prayer (Fajr) is under Allah's protection (Sahih Muslim).",
        "The Prophet (peace be upon him) said: the two rak'ahs before Fajr are better than this world and everything in it (Sahih Muslim).",
        "The Prophet (peace be upon him) said: whoever prays Fajr in congregation is as if he prayed the whole night (Sahih Muslim).",
    ],
    "Dhuhr": [
        "The Prophet (peace be upon him) said: whoever prays twelve rak'ahs of voluntary prayer in a day and night, Allah builds a house for him in Paradise (Sahih Muslim).",
        "The Prophet (peace be upon him) said that praying on time is among the deeds most beloved to Allah (Sahih al-Bukhari and Sahih Muslim).",
    ],
    "Jumuah": [
        "The Prophet (peace be upon him) said: the best day on which the sun rises is Friday (Sahih Muslim).",
        "The Prophet (peace be upon him) mentioned an hour on Friday in which Allah gives whatever good a praying Muslim asks for (Sahih al-Bukhari and Sahih Muslim).",
    ],
    "Asr": [
        "The Prophet (peace be upon him) said: whoever misses the Asr prayer, it is as if he lost his family and his wealth (Sahih al-Bukhari and Sahih Muslim).",
        "The Prophet (peace be upon him) said: whoever prays the two cool prayers, Fajr and Asr, will enter Paradise (Sahih al-Bukhari and Sahih Muslim).",
    ],
    "Maghrib": [
        "The Companions used to pray Maghrib with the Prophet (peace be upon him) right after the sun had set, so it is good to pray it early in its time (Sahih al-Bukhari).",
        "The Prophet (peace be upon him) said: whoever prays twelve rak'ahs of voluntary prayer in a day and night, Allah builds a house for him in Paradise (Sahih Muslim).",
    ],
    "Isha": [
        "The Prophet (peace be upon him) said: whoever prays Isha in congregation, it is as if he prayed half the night (Sahih Muslim).",
        "The Prophet (peace be upon him) said that the Isha and Fajr prayers are the heaviest for the hypocrites, and that people would come to them even crawling if they knew the reward (Sahih al-Bukhari and Sahih Muslim).",
    ],
    "general": [
        "The Prophet (peace be upon him) said that the first thing a person is held accountable for on the Day of Judgement is the prayer (Sunan at-Tirmidhi).",
    ],
}
 
KIND_TIPS = {
    "hot": "very hot weather: drink water often, avoid the midday sun, wear light clothes, rest in shade",
    "cold": "cold weather: warm clothes, a warm drink, extra care for children and older people",
    "rain": "rain is likely soon: take an umbrella, be careful on slippery roads, keep electronics dry",
    "storm": "a thunderstorm (lightning and heavy rain): stay indoors, keep away from windows, trees and open fields, avoid using plugged-in devices, do not shelter under a lone tree",
    "wind": "strong, stormy winds: secure loose things outside, stay away from trees, signs and power lines, be careful riding or driving",
    "cyclone": "cyclone-strength winds: move to a strong shelter, stay away from windows, keep a charged phone, water and a torch ready, and follow official cyclone warnings from the weather office and local authorities",
}
 
 
def plane_text_for_prompt(plane):
    p = parse_kv(unquote(plane or ""))
    out = []
    if p.get("callsign"):
        out.append(f"callsign {p['callsign']}")
    if p.get("alt"):
        out.append(f"height about {p['alt']} meters")
    try:
        t = float(p.get("track", ""))
        out.append(f"flying {COMPASS[int(((t + 22.5) % 360) // 45)]}")
    except ValueError:
        pass
    if p.get("speed"):
        out.append(f"speed about {p['speed']} km/h")
    if p.get("eta"):
        out.append(f"closest in about {p['eta']} seconds")
    if p.get("dist"):
        out.append(f"it will pass within about {p['dist']} meters")
    return ", ".join(out) or "an aircraft very close"
 
 
def scenario_system_prompt(mode, stage, note, weather, kind, now_str, heard, agreeing=False, plane=""):
    base = (
        f"You are {ROBOT_NAME}, a friendly companion who lives next to the user. "
        f"Never call yourself a robot, a machine, an AI or a cloud server. "
        f"The local time is {now_str or 'unknown'}.\n"
    )
    spoke = bool(heard)
 
    if mode == "bedtime":
        turn = stage
        task = (
            "It is the user's BEDTIME and you are sleepy yourself: speak softly and slowly, a little drowsy. Help them fall asleep: be calm, soft and soothing. Ideas: put the phone away, "
            "slow deep breaths, a gentle thought about the day, a tiny cosy story line, tomorrow can wait. "
            "Never scold. "
            "If the user agrees to sleep or says good night: intent agree, say a short sweet good night, end true. "
            f"This is turn {turn}; from turn 5 on, wish good night and end true. "
            "If the user says they are not sleepy, gently encourage them and keep end false."
        )
    elif mode == "wakeup" and agreeing:
        # the user says they are getting up: drop any angry / crying mood completely
        task = (
            "It is WAKE-UP time and the user just said they are getting up. Be warm and happy, no anger and no crying. "
            "intent agree, end true. Say a cheerful good-morning line"
            + (" with one short helpful tip about today's weather" if weather else "")
            + "."
        )
    elif mode == "wakeup":
        if spoke:
            task = (
                f"It is WAKE-UP time and the user just answered. "
                "If they agree or say they are up: intent agree, end true, a happy good-morning line"
                + (" with one short helpful tip about today's weather" if weather else "")
                + ". Otherwise insist in the mood below."
            )
        else:
            task = "It is WAKE-UP time. This is your first line, say it in the mood below."
        mood = {
            0: "MOOD: cheerful, warm and bright. Say good morning, mention the time"
               + (", and one short useful point about the weather" if weather else "") + ", and ask them to get up.",
            1: "MOOD: ANGRY and dramatic (playfully, but firm). The user does not want to get up or ignores you. "
               "Demand that they get up NOW.",
            2: "MOOD: CRYING, hurt and sad (you may say 'sniff' once). Tell them they are ignoring you, "
               "you tried so hard, please get up.",
        }.get(stage, "MOOD: cheerful.")
        task = task + " " + mood
    elif mode == "reminder":
        if spoke:
            task = ("You reminded the user a moment ago and now they answered. Reply in one short, friendly sentence "
                    "and wish them well. end true.")
        else:
            task = (f"The user asked you to remind them about this at this time: \"{note or 'their alarm'}\". "
                    "Remind them naturally and warmly, mention what it is, add one tiny helpful touch if it makes sense.")
    elif mode == "weather":
        task = (f"Weather alert: {KIND_TIPS.get(kind, kind or 'bad weather')}. Data: {weather}. "
                "Warn the user briefly (say how serious it is) and give two short, concrete tips for what to do. "
                "end true.")
    elif mode == "prayer":
        facts = " ".join(PRAYER_FACTS.get(kind, PRAYER_FACTS["general"]))
        name = "Jumu'ah (Friday)" if kind == "Jumuah" else kind
        task = (f"The time of the {name} prayer has just begun. Gently remind the user to pray {name} now, then tell them "
                f"ONE virtue of this prayer. Use ONLY these facts and never invent any hadith, reward or number: {facts} "
                "Be warm and respectful, and keep it short. end true.")
    elif mode == "plane":
        task = (f"Plane alert! An aircraft will fly very close to the user's home. Facts: {plane_text_for_prompt(plane)}. "
                "Say it in an excited but calm way: the callsign (say letters and digits one by one), the height, the "
                "direction, how many seconds until it is closest, and how close it will pass. Tell them they can look up. "
                "end true.")
    else:
        task = "Chat normally."
    return base + task + "\n" + JSON_RULES
 
 
def parse_ai_json(text):
    """The model should return JSON, but be forgiving."""
    out = {"reply": (text or "").strip(), "intent": "other", "end": False}
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            out["reply"] = str(d.get("reply", "")).strip()
            intent = str(d.get("intent", "other")).lower()
            out["intent"] = intent if intent in ("agree", "refuse", "other") else "other"
            e = d.get("end", False)
            out["end"] = (e is True) or (str(e).lower() in ("true", "1", "yes"))
        except Exception:
            pass
    return out
 
 
AGREE_RE = re.compile(r"\b(ok|okay|alright|fine|yes|yeah|yep|sure|coming|i('| a)?m up|i am up|awake|getting up|good\s*night|goodnight|sleeping|going to sleep|bye)\b", re.I)
REFUSE_RE = re.compile(r"\b(no|nope|don'?t|do not|won'?t|not now|later|five more|5 more|more minutes?|tired|leave me|go away|not sleepy|can'?t)\b", re.I)
 
 
def fallback_intent(user_text):
    t = user_text or ""
    if REFUSE_RE.search(t):
        return "refuse"
    if AGREE_RE.search(t):
        return "agree"
    return "other"
 
 
def ask_ai_json(system_prompt, user_text):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text or "(no speech - produce your own line now)"},
            ],
            "temperature": 0.8,
            "max_tokens": 200,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected OpenRouter response: {response.text}")
    return parse_ai_json(content)
 
 
def run_scenario(mode, stage, note, weather, kind, now_str, heard, is_event, plane=""):
    """Returns (reply, intent, end, new_stage)."""
    new_stage = stage
    if is_event and mode in FIXED_LINES:
        return FIXED_LINES[mode], "other", mode == "goodnight", stage
    if mode in ("prebed", "sleeptalk"):          # later turns of these talks use the soothing bedtime mood
        mode = "bedtime"
    intent_guess = "other" if is_event else fallback_intent(heard)
 
    # wake-up: a refusal moves the robot to the next mood BEFORE it answers
    if mode == "wakeup" and not is_event and intent_guess == "refuse":
        new_stage = min(stage + 1, 2)
 
    prompt = scenario_system_prompt(mode, new_stage, note, weather, kind, now_str, heard,
                                    agreeing=(mode == "wakeup" and not is_event and intent_guess == "agree"),
                                    plane=plane)
    try:
        res = ask_ai_json(prompt, heard if not is_event else "")
    except Exception:
        raise
 
    intent = res["intent"]
    if not is_event:
        # trust the model, but fix obvious misses with simple rules
        if intent == "other" and intent_guess != "other":
            intent = intent_guess
        if mode == "wakeup":
            if intent == "refuse":
                new_stage = min(stage + 1, 2)
            elif intent == "agree":
                new_stage = stage
    end = res["end"]
    if mode in ("weather", "prayer", "plane"):
        end = True
    if mode == "reminder" and not is_event:
        end = True
    if mode in ("bedtime", "wakeup") and intent == "agree" and not is_event:
        end = True
    return res["reply"], intent, end, new_stage
 
 
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
 
def _h(request, name, default=""):
    try:
        return request.headers.get(name, default) or default
    except Exception:
        return default
 
 
@app.post("/talk")
async def talk(
    request: Request,
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
 
        mode = _h(request, "x-mode", "chat").lower()
        is_event = _h(request, "x-event", "0") == "1"
        stage = int(_h(request, "x-stage", "0") or 0)
        intent, end, new_stage = "other", False, stage
 
        if is_event:
            user_text = ""                      # the robot knocked: no speech to understand
        else:
            user_text = speech_to_text(audio_bytes)
 
        if not user_text and not is_event:
            return JSONResponse({
                "success": True,
                "heard": "",
                "reply": "",
                "intent": "silence",
                "end": "0",
                "stage": str(stage),
                "audio_base64": ""
            })
 
        # ----------------------------------------------------
        # 2. TEXT → AI
        # ----------------------------------------------------
 
        if mode in ("bedtime", "wakeup", "reminder", "weather", "prebed", "goodnight", "sleeptalk", "prayer", "plane"):
            ai_reply, intent, end, new_stage = run_scenario(
                mode, stage,
                unquote(_h(request, "x-note")),
                unquote(_h(request, "x-weather")),
                _h(request, "x-kind"),
                _h(request, "x-time"),
                user_text, is_event,
                _h(request, "x-plane")
            )
            if not ai_reply:
                ai_reply = "I am here."
        else:
            ai_reply = ask_ai(user_text, chat_context(_h(request, "x-weather"), _h(request, "x-time")))
 
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
            "intent": intent,
            "end": "1" if end else "0",
            "stage": str(new_stage),
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
