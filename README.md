# AI Study Buddy (3 files)

Files: `app.py`, `index.html`, `README.md`. Keep `app.py` and `index.html` in the same folder.

## Run it
Double-click `app.py` (or run `python app.py`). The first run installs what it needs, then Study Buddy opens in its own window.
If an app window cannot be set up, it opens in your browser instead. Close the black window to stop it.

## First time
Paste your free Gemini key in the box (get it at https://aistudio.google.com/apikey). It is saved in `key.txt` next to `app.py`. Do not share that file.

## What it does
Upload a PDF or text file, then: Summary, Ask (answers from your notes only), Quiz (5 MCQs), Flashcards.
Notes are kept in memory only, so upload again after closing.

## Make a real .exe (optional, Windows)
Open a terminal in this folder and run:

    pip install pyinstaller pywebview fastapi uvicorn python-multipart requests pypdf scikit-learn
    pyinstaller --noconfirm --windowed --name StudyBuddy --add-data "index.html;." --collect-submodules uvicorn --hidden-import multipart --hidden-import python_multipart app.py

Your app is `dist\StudyBuddy\StudyBuddy.exe` (keep that whole folder together). In the .exe your key is saved in `%APPDATA%\StudyBuddy`.
