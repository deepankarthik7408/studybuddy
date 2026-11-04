"""AI Study Buddy: summary, cheat sheet, questions, quiz, mock exam, flashcards.

Double-click this file (or run: python app.py) to open it as an app window.
To use it on a phone, put it online (see README.md) and install it from the browser.
"""
import io
import json
import os
import re
import sys
import tempfile
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import importlib
import importlib.util
import socket
import struct
import subprocess
import threading
import webbrowser
import zlib


def _have(m):
    return importlib.util.find_spec(m) is not None


if (not getattr(sys, "frozen", False) and os.getenv("HOSTED") != "1"
        and not (all(_have(m) for m in ("fastapi", "uvicorn", "requests", "pypdf", "sklearn"))
                 and (_have("python_multipart") or _have("multipart")))):
    print("First run: installing what Study Buddy needs. This can take a few minutes...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn",
                           "python-multipart", "requests", "pypdf", "scikit-learn"])
    importlib.invalidate_caches()

import requests
import uvicorn
from urllib.parse import urlparse
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HOSTED = os.getenv("HOSTED") == "1"  # set on a public website: keys then live in each visitor's browser
REQ_CFG = ContextVar("req_cfg", default=None)
HOSTED_OPENAI_HOSTS = {"api.openai.com", "api.groq.com", "openrouter.ai"}
if getattr(sys, "frozen", False):  # running as the built StudyBuddy.exe
    RES = Path(sys._MEIPASS)                                   # bundled files (index.html)
    DATA = Path(os.getenv("APPDATA", str(Path.home()))) / "StudyBuddy"  # your keys and reports
else:
    RES = Path(__file__).parent
    DATA = Path(tempfile.gettempdir()) if HOSTED else RES
DATA.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA / "settings.json"   # holds your API keys: do not share this file
REPORTS_FILE = DATA / "reports.jsonl"    # bug reports from the Report button
MAX_CONTEXT_CHARS = 30000
PROVIDERS = ["gemini", "claude", "openai"]
DEFAULTS = {
    "mode": "auto",
    "gemini": {"key": "", "model": "gemini-3.8-flash"},
    "claude": {"key": "", "model": "claude-haiku-5-5"},
    "openai": {"key": "", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
}
GEMINI_FALLBACKS = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
LAST = {"provider": None}
DOCS: dict[str, dict] = {}

app = FastAPI(title="AI Study Buddy")


@app.middleware("http")
async def read_client_config(request, call_next):
    raw = request.headers.get("x-ai-config")  # hosted mode: the visitor's own keys
    if raw:
        try:
            REQ_CFG.set(json.loads(raw))
        except ValueError:
            pass
    return await call_next(request)


# ---------- settings ----------
def load_settings(include_env: bool = True) -> dict:
    s = json.loads(json.dumps(DEFAULTS))
    if HOSTED:  # nothing is stored on the server; use what this visitor's browser sent
        c = REQ_CFG.get() or {}
        if c.get("mode") in ["auto"] + PROVIDERS:
            s["mode"] = c["mode"]
        for p in PROVIDERS:
            for k, v in (c.get(p) or {}).items():
                if k in s[p] and isinstance(v, str) and (k == "key" or v.strip()):
                    if k == "base_url" and urlparse(v.strip()).hostname not in HOSTED_OPENAI_HOSTS:
                        continue  # stops visitors pointing the server at other machines
                    s[p][k] = v.strip()
        return s
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            s["mode"] = saved.get("mode", s["mode"])
            for p in PROVIDERS:
                for k, v in saved.get(p, {}).items():
                    if k in s[p] and (k == "key" or v):
                        s[p][k] = v
        except (OSError, json.JSONDecodeError):
            pass
    if include_env and not s["gemini"]["key"]:
        s["gemini"]["key"] = os.getenv("GEMINI_API_KEY", "")
    return s


# ---------- AI providers ----------
class ProviderError(Exception):
    pass


def post_retry(url, payload, headers):
    res = None
    for _ in range(2):  # retry once when the service is overloaded
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=120)
        except requests.RequestException as e:
            raise ProviderError(f"network error: {e}")
        if res.status_code not in (500, 503, 504):
            break
        time.sleep(2)
    return res


def call_gemini(cfg, system, user, max_tokens):
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        # thinking models spend part of this budget, so keep it generous
        "generationConfig": {"maxOutputTokens": max(max_tokens, 8000)},
    }
    last = "unknown error"
    for model in dict.fromkeys([cfg["model"]] + GEMINI_FALLBACKS):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        res = post_retry(url, payload, {"x-goog-api-key": cfg["key"]})
        if res.ok:
            try:
                parts = res.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts)
            except (KeyError, IndexError):
                last = "empty answer"
                continue
        last = f"{res.status_code} {res.text[:150]}"
        if res.status_code in (401, 403):
            break
    raise ProviderError("Gemini: " + last)


def call_claude(cfg, system, user, max_tokens):
    res = post_retry(
        "https://api.anthropic.com/v1/messages",
        {"model": cfg["model"], "max_tokens": max_tokens, "system": system,
         "messages": [{"role": "user", "content": user}]},
        {"x-api-key": cfg["key"], "anthropic-version": "2023-06-01"},
    )
    if not res.ok:
        raise ProviderError(f"Claude: {res.status_code} {res.text[:150]}")
    return "".join(b.get("text", "") for b in res.json().get("content", []) if b.get("type") == "text")


def call_openai(cfg, system, user, max_tokens):
    res = post_retry(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        {"model": cfg["model"],
         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        {"Authorization": "Bearer " + cfg["key"]},
    )
    if not res.ok:
        raise ProviderError(f"OpenAI: {res.status_code} {res.text[:150]}")
    try:
        return res.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise ProviderError("OpenAI: empty answer")


FUNCS = {"gemini": call_gemini, "claude": call_claude, "openai": call_openai}


def ask_llm(system: str, user: str, max_tokens: int = 1500) -> str:
    s = load_settings()
    names = PROVIDERS if s["mode"] == "auto" else [s["mode"]]
    names = [n for n in names if s[n]["key"]]
    if not names:
        raise HTTPException(400, "No API key saved for the chosen AI. Open the Settings tab.")
    errors = []
    for n in names:  # auto mode: first working provider wins
        try:
            text = FUNCS[n](s[n], system, user, max_tokens)
            LAST["provider"] = n
            return text
        except ProviderError as e:
            errors.append(str(e))
    raise HTTPException(503, " | ".join(errors)[:400])


def ask_json(system: str, user: str, tokens: int = 4000):
    raw = ask_llm(system + " Respond with valid JSON only: no markdown fences, no extra text.", user, tokens)
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    try:
        return json.loads(m.group(0))
    except (AttributeError, json.JSONDecodeError):
        raise HTTPException(502, "The AI sent a broken answer. Click the button again.")


# ---------- documents ----------
def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def get_doc(doc_id: str) -> dict:
    if doc_id not in DOCS:
        raise HTTPException(404, "Document not found. Upload it again.")
    return DOCS[doc_id]


def notes(doc_id: str) -> str:
    return get_doc(doc_id)["text"][:MAX_CONTEXT_CHARS]


def top_chunks(doc: dict, question: str, k: int = 4) -> list[str]:
    scores = cosine_similarity(doc["vectorizer"].transform([question]), doc["matrix"])[0]
    return [doc["chunks"][i] for i in scores.argsort()[::-1][:k]]


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


def clean_qa(items) -> list[dict]:
    return [q for q in items if isinstance(q, dict) and q.get("question")] if isinstance(items, list) else []


MCQ_SPEC = ("objects with keys: question, options (list of exactly 4 strings), "
            "answer (the index 0-3 of the correct option, as a number), explanation (one sentence)")


# ---------- routes ----------
@app.get("/")
def home():
    return FileResponse(RES / "index.html")


MANIFEST = {
    "name": "Study Buddy", "short_name": "Study Buddy", "start_url": "/", "display": "standalone",
    "background_color": "#16213E", "theme_color": "#16213E",
    "icons": [{"src": f"/icon-{n}.png", "sizes": f"{n}x{n}", "type": "image/png", "purpose": "any maskable"}
              for n in (192, 512)],
}
SW_JS = """const CACHE = "sb-v1";
self.addEventListener("install", (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(["/"]))); self.skipWaiting(); });
self.addEventListener("activate", (e) => e.waitUntil(clients.claim()));
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.pathname.startsWith("/api/")) return;
  e.respondWith(fetch(e.request)
    .then((r) => { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request)));
});
"""
ICONS: dict[int, bytes] = {}


def make_png(n: int) -> bytes:
    raw = bytearray()
    for y in range(n):
        raw.append(0)
        for x in range(n):
            dx, dy = x - n / 2, y - n / 2
            if dx * dx + dy * dy < (0.26 * n) ** 2:
                c = (14, 124, 134)
            elif abs(dx) < 0.22 * n and 0.30 * n < dy < 0.36 * n:
                c = (255, 210, 63)
            else:
                c = (22, 33, 62)
            raw += bytes(c)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)

    return (bytes([137, 80, 78, 71, 13, 10, 26, 10]) + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse(MANIFEST, media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return Response(SW_JS, media_type="application/javascript")


@app.get("/icon-{size}.png")
def icon(size: int):
    if size not in (192, 512):
        raise HTTPException(404)
    if size not in ICONS:
        ICONS[size] = make_png(size)
    return Response(ICONS[size], media_type="image/png")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 15_000_000:
        raise HTTPException(413, "File is too big (15 MB limit).")
    text = extract_text(file.filename, data)
    if len(text.strip()) < 50:
        raise HTTPException(400, "Could not read any text from this file.")
    chunks = chunk_text(text)
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_id = uuid.uuid4().hex
    while len(DOCS) >= 40:  # keep memory bounded on a shared server
        DOCS.pop(next(iter(DOCS)))
    DOCS[doc_id] = {"name": file.filename, "text": text, "chunks": chunks,
                    "vectorizer": vectorizer, "matrix": vectorizer.fit_transform(chunks)}
    return {"doc_id": doc_id, "name": file.filename, "chunks": len(chunks)}


class Ask(BaseModel):
    doc_id: str
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    context = "\n\n---\n\n".join(top_chunks(get_doc(body.doc_id), body.question))
    return {"answer": ask_llm(
        "You are a study assistant. Answer using ONLY the notes provided. "
        "If the notes do not contain the answer, say so. Be clear and concise.",
        f"NOTES:\n{context}\n\nQUESTION: {body.question}")}


class Gen(BaseModel):
    doc_id: str
    count: int = 5
    kind: str = "mcq"


@app.post("/api/summary")
def summary(body: Gen):
    return {"summary": ask_llm(
        "You are a study assistant. Summarise the notes as short headed sections "
        "with bullet points a student can revise from in 5 minutes.", notes(body.doc_id), 3000)}


@app.post("/api/quiz")
def quiz(body: Gen):
    n = max(1, min(body.count, 20))
    qs = clean_mcqs(ask_json(f"You write multiple-choice exam questions from study notes. "
                             f"Return a JSON list of {n} {MCQ_SPEC}.", notes(body.doc_id), 6000))
    if not qs:
        raise HTTPException(502, "No usable questions came back. Click the button again.")
    return {"questions": qs}


@app.post("/api/flashcards")
def flashcards(body: Gen):
    n = max(1, min(body.count, 30))
    cards = [c for c in ask_json(
        f"You make revision flashcards from study notes. Return a JSON list of {n} objects "
        "with keys: front (a question or term), back (a short answer).", notes(body.doc_id), 5000)
        if isinstance(c, dict) and c.get("front") and c.get("back")]
    if not cards:
        raise HTTPException(502, "No usable cards came back. Click the button again.")
    return {"cards": cards}


@app.post("/api/cheatsheet")
def cheatsheet(body: Gen):
    data = ask_json(
        "You make a last-minute revision cheat sheet from study notes: dense, scannable, nothing fluffy. "
        "Return a JSON object with keys: sections (list of {title, points: list of short strings}), "
        "key_terms (list of {term, meaning}), formulas_and_rules (list of strings, may be empty), "
        "remember (list of 5 to 8 must-not-forget lines).", notes(body.doc_id), 6000)
    if not isinstance(data, dict):
        raise HTTPException(502, "The AI sent a broken answer. Click the button again.")
    return data


KINDS = {
    "2": ("2-mark short-answer questions", "question, answer (2 to 3 lines)"),
    "5": ("5-mark questions", "question, answer (about 5 key points)"),
    "10": ("10-mark long-answer questions", "question, answer (structured: intro, main points, conclusion)"),
    "important": ("of the most important questions likely to be asked in the exam, most important first",
                  "question, marks (a number), why (one short line on why it matters)"),
}


@app.post("/api/questions")
def questions(body: Gen):
    n = max(1, min(body.count, 20))
    if body.kind == "mcq":
        items = clean_mcqs(ask_json(f"You write multiple-choice exam questions from study notes. "
                                    f"Return a JSON list of {n} {MCQ_SPEC}.", notes(body.doc_id), 8000))
    elif body.kind in KINDS:
        desc, keys = KINDS[body.kind]
        items = clean_qa(ask_json(
            f"You are an exam-paper setter. From the notes write {n} {desc}. "
            f"Return a JSON list of objects with keys: {keys}.", notes(body.doc_id), 8000))
    else:
        raise HTTPException(400, "Unknown question type.")
    if not items:
        raise HTTPException(502, "No usable questions came back. Click the button again.")
    return {"items": items}


class Mock(BaseModel):
    doc_id: str
    mcq: int = 10
    two: int = 5
    five: int = 3
    ten: int = 2


@app.post("/api/mock")
def mock(body: Mock):
    c = {k: max(0, min(getattr(body, k), 30)) for k in ("mcq", "two", "five", "ten")}
    if not sum(c.values()):
        raise HTTPException(400, "Choose at least one question.")
    parts = []
    if c["mcq"]: parts.append(f'"mcq": {c["mcq"]} {MCQ_SPEC}')
    if c["two"]: parts.append(f'"two": {c["two"]} 2-mark questions as objects with keys question, answer (2 to 3 lines)')
    if c["five"]: parts.append(f'"five": {c["five"]} 5-mark questions as objects with keys question, answer (about 5 key points)')
    if c["ten"]: parts.append(f'"ten": {c["ten"]} 10-mark questions as objects with keys question, answer (structured: intro, main points, conclusion)')
    data = ask_json("You are an exam-paper setter. Write a mock exam from the notes, covering different topics, "
                    "no repeated questions. Return a JSON object with these keys: " + "; ".join(parts) + ".",
                    notes(body.doc_id), 14000)
    if not isinstance(data, dict):
        raise HTTPException(502, "The AI sent a broken answer. Click the button again.")
    return {"mcq": clean_mcqs(data.get("mcq", [])), "two": clean_qa(data.get("two", [])),
            "five": clean_qa(data.get("five", [])), "ten": clean_qa(data.get("ten", []))}


class Report(BaseModel):
    tab: str = ""
    message: str
    context: str = ""


@app.post("/api/report")
def report(body: Report):
    line = {"time": datetime.now().isoformat(timespec="seconds"), "tab": body.tab[:40],
            "ai": LAST["provider"], "message": body.message[:2000], "context": body.context[:4000]}
    with open(REPORTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return {"ok": True}


class SettingsIn(BaseModel):
    mode: str = "auto"
    gemini: dict = {}
    claude: dict = {}
    openai: dict = {}
    clear: list[str] = []


@app.get("/api/settings")
def get_settings():
    if HOSTED:
        return {"hosted": True}
    s = load_settings()
    out = {"hosted": False, "mode": s["mode"]}
    for p in PROVIDERS:
        k = s[p]["key"]
        out[p] = {"has_key": bool(k), "hint": k[-4:] if k else "", "model": s[p]["model"]}
    out["openai"]["base_url"] = s["openai"]["base_url"]
    return out


@app.post("/api/settings")
def save_settings(body: SettingsIn):
    if HOSTED:
        raise HTTPException(400, "Settings are saved in your browser on this site.")
    s = load_settings(include_env=False)
    if body.mode in ["auto"] + PROVIDERS:
        s["mode"] = body.mode
    for p in PROVIDERS:
        for k in s[p]:
            v = str(getattr(body, p).get(k, "")).strip()
            if v:
                s[p][k] = v
        if p in body.clear:
            s[p]["key"] = ""
    SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return {"ok": True}


@app.post("/api/test")
def test():
    reply = ask_llm("Reply with the single word OK.", "Say OK.", 20)
    return {"ok": True, "provider": LAST["provider"], "reply": reply.strip()[:40]}


# ---------- open as an app window (local use) ----------
def main():
    port = 8765  # a steady address, so a browser-installed copy keeps working
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        port = 0
    if port == 0:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    probe.close()
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
        if not getattr(sys, "frozen", False) and not skip.exists():
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
            webview.create_window("Study Buddy", url, width=1100, height=820)
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
