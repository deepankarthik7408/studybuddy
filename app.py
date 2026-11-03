"""AI Study Buddy. Double-click this file (or run: python app.py) and it opens as an app window.
The first run installs what it needs. Your free Gemini key is pasted inside the app."""
import importlib
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)


def _have(m):
    return importlib.util.find_spec(m) is not None


if not FROZEN and not (all(_have(m) for m in ("fastapi", "uvicorn", "requests", "pypdf", "sklearn"))
                       and (_have("python_multipart") or _have("multipart"))):
    print("First run: installing what Study Buddy needs. This can take a few minutes...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn",
                           "python-multipart", "requests", "pypdf", "scikit-learn"])
    importlib.invalidate_caches()

import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if FROZEN:
    RES = Path(sys._MEIPASS)
    DATA = Path(os.getenv("APPDATA", str(Path.home()))) / "StudyBuddy"
else:
    RES = DATA = Path(__file__).parent
DATA.mkdir(parents=True, exist_ok=True)
KEY_FILE = DATA / "key.txt"  # your Gemini key is saved here, on your computer only
MODELS = list(dict.fromkeys([os.getenv("MODEL", "gemini-3.8-flash"), "gemini-2.5-flash", "gemini-3.5-flash-lite"]))
MAX_CHARS = 30000
DOCS: dict[str, dict] = {}
app = FastAPI(title="AI Study Buddy")


# ---------- Gemini ----------
def get_key() -> str:
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return os.getenv("GEMINI_API_KEY", "")


def ask_llm(system: str, user: str) -> str:
    key = get_key()
    if not key:
        raise HTTPException(400, "Paste your Gemini key first.")
    payload = {"system_instruction": {"parts": [{"text": system}]},
               "contents": [{"role": "user", "parts": [{"text": user}]}],
               "generationConfig": {"maxOutputTokens": 8000}}
    last = "unknown error"
    for model in MODELS:  # if one model is busy (503) or limited (429), try the next
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for _ in range(2):
            try:
                res = requests.post(url, json=payload, headers={"x-goog-api-key": key}, timeout=120)
            except requests.RequestException as e:
                raise HTTPException(503, f"Network problem: {e}")
            if res.status_code not in (500, 503, 504):
                break
            time.sleep(2)
        if res.ok:
            try:
                return "".join(p.get("text", "") for p in res.json()["candidates"][0]["content"]["parts"])
            except (KeyError, IndexError):
                last = "empty answer"
                continue
        last = f"{res.status_code} {res.text[:150]}"
        if res.status_code in (401, 403):
            break
    raise HTTPException(503, "Gemini problem: " + last)


def ask_json(system: str, user: str):
    raw = ask_llm(system + " Respond with valid JSON only: no markdown fences, no extra text.", user)
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    try:
        return json.loads(m.group(0))
    except (AttributeError, json.JSONDecodeError):
        raise HTTPException(502, "The AI sent a broken answer. Click the button again.")


# ---------- documents ----------
def get_doc(doc_id: str) -> dict:
    if doc_id not in DOCS:
        raise HTTPException(404, "Document not found. Upload it again.")
    return DOCS[doc_id]


def clean_mcqs(items) -> list[dict]:
    out = []
    for q in items if isinstance(items, list) else []:
        try:
            opts = [str(o) for o in q["options"]][:4]
            a = q["answer"]
            if isinstance(a, str):
                a = a.strip()
                a = "ABCD".index(a.upper()[0]) if a[:1].upper() in "ABCD" and not a.isdigit() else int(a)
            if len(opts) == 4 and 0 <= int(a) < 4:
                out.append({"question": str(q["question"]), "options": opts, "answer": int(a),
                            "explanation": str(q.get("explanation", ""))})
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return out


# ---------- routes ----------
@app.get("/")
def home():
    return FileResponse(RES / "index.html")


class KeyIn(BaseModel):
    key: str


@app.get("/api/key")
def key_status():
    return {"has_key": bool(get_key())}


@app.post("/api/key")
def save_key(body: KeyIn):
    KEY_FILE.write_text(body.key.strip(), encoding="utf-8")
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if file.filename.lower().endswith(".pdf"):
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    if len(text.strip()) < 50:
        raise HTTPException(400, "Could not read any text from this file.")
    flat = re.sub(r"\s+", " ", text).strip()
    chunks = [flat[i:i + 900] for i in range(0, len(flat), 750)]  # 900 characters, 150 overlap
    vec = TfidfVectorizer(stop_words="english")
    doc_id = uuid.uuid4().hex[:8]
    DOCS[doc_id] = {"text": text, "chunks": chunks, "vec": vec, "matrix": vec.fit_transform(chunks)}
    return {"doc_id": doc_id, "name": file.filename, "chunks": len(chunks)}


class Ask(BaseModel):
    doc_id: str
    question: str = ""
    count: int = 5


@app.post("/api/ask")
def ask(body: Ask):
    d = get_doc(body.doc_id)
    scores = cosine_similarity(d["vec"].transform([body.question]), d["matrix"])[0]
    context = "\n\n---\n\n".join(d["chunks"][i] for i in scores.argsort()[::-1][:4])
    return {"answer": ask_llm("You are a study assistant. Answer using ONLY the notes provided. "
                              "If the notes do not contain the answer, say so. Be clear and concise.",
                              f"NOTES:\n{context}\n\nQUESTION: {body.question}")}


@app.post("/api/summary")
def summary(body: Ask):
    return {"summary": ask_llm("You are a study assistant. Summarise the notes as short headed sections "
                               "with bullet points a student can revise from in 5 minutes.",
                               get_doc(body.doc_id)["text"][:MAX_CHARS])}


@app.post("/api/quiz")
def quiz(body: Ask):
    qs = clean_mcqs(ask_json(
        f"You write multiple-choice exam questions from study notes. Return a JSON list of {max(1, min(body.count, 20))} "
        "objects with keys: question, options (list of exactly 4 strings), answer (index 0-3 of the correct "
        "option, as a number), explanation (one sentence).", get_doc(body.doc_id)["text"][:MAX_CHARS]))
    if not qs:
        raise HTTPException(502, "No usable questions came back. Click the button again.")
    return {"questions": qs}


@app.post("/api/flashcards")
def flashcards(body: Ask):
    cards = [c for c in ask_json(
        f"You make revision flashcards from study notes. Return a JSON list of {max(1, min(body.count, 30))} "
        "objects with keys: front (a question or term), back (a short answer).",
        get_doc(body.doc_id)["text"][:MAX_CHARS]) if isinstance(c, dict) and c.get("front") and c.get("back")]
    if not cards:
        raise HTTPException(502, "No usable cards came back. Click the button again.")
    return {"cards": cards}


# ---------- open as an app ----------
def main():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    skip = DATA / "no-webview"  # remembers that the app window could not be installed
    webview = None
    try:
        import webview
    except ImportError:
        if not FROZEN and not skip.exists():
            print("Setting up the app window (one time)...")
            subprocess.call([sys.executable, "-m", "pip", "install", "pywebview"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            importlib.invalidate_caches()
            try:
                import webview
            except ImportError:
                skip.write_text("x")
    if webview:
        try:
            webview.create_window("Study Buddy", url, width=1000, height=820)
            webview.start()
            return
        except Exception:
            pass
    webbrowser.open(url)  # fallback: your browser
    print("Study Buddy is running at", url, "- close this window to stop it.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
