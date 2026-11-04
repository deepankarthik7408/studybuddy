# AI Study Buddy (3 files: app.py, index.html, README.md)

Features: Summary, Cheat sheet (printable), Ask your notes, Questions (important / MCQ / 2, 5, 10 mark),
Quiz, Mock exam (timer + scoring), Flashcards, Settings (choose Gemini / Claude / OpenAI or Auto), Report a bug.

## A) Use it on your PC
Keep `app.py` and `index.html` in one folder. Double-click `app.py` (or run `python app.py`).
The first run installs what it needs and opens Study Buddy in its own window (or your browser).
Open the **Settings** tab, paste a free Gemini key (https://aistudio.google.com/apikey), Save, then Test connection.
Your keys are saved in `settings.json` next to `app.py`. Never upload or share that file.
Bug reports are saved in `reports.jsonl`.

## B) Install it on a phone or any device (put it online first)
1. Put `app.py`, `index.html` and `README.md` in a GitHub repository (do NOT upload settings.json or reports.jsonl).
2. On render.com create a **Web Service** from that repo and set:
   - Build command: `pip install fastapi uvicorn python-multipart requests pypdf scikit-learn`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Environment variable: `HOSTED` = `1`
3. Open the https address Render gives you.
4. Install it:
   - Android / PC (Chrome, Edge): tap the **Install app** button at the top of the page (or the install icon in the address bar).
   - iPhone: in Safari tap Share, then **Add to Home Screen**.

Online mode: nothing is stored on the server. Each person pastes their own API key in Settings and it stays in their own browser.
Uploaded notes live in server memory only, so they disappear when the server restarts or sleeps (free plans sleep when idle).

## Make a .exe (optional, Windows)
    pip install pyinstaller pywebview fastapi uvicorn python-multipart requests pypdf scikit-learn
    pyinstaller --noconfirm --windowed --name StudyBuddy --add-data "index.html;." --collect-submodules uvicorn --hidden-import multipart --hidden-import python_multipart app.py
Result: `dist\StudyBuddy\StudyBuddy.exe` (keep the whole folder). Keys are saved in `%APPDATA%\StudyBuddy`.
