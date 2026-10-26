# Unlox — Domain-Specific RAG Chatbot for PDF Question Answering

## Run in VS Code (Windows)
1. Extract this folder.
2. Open the folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run `.venv\Scripts\activate.bat`, then continue.
5. Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.
6. Upload PDF(s) in the sidebar, click **Process Documents**, then ask questions.

## Optional natural-language answer generation
Without an API key, the app retrieves and displays relevant passages with source pages. To enable Gemini generation, set `GEMINI_API_KEY` in your environment before starting Streamlit. Never put the key in source code or commit it to GitHub.

PowerShell (current terminal only):
```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
streamlit run app.py
```

## Project files
- `app.py`: Streamlit UI with upload, Process Documents, chat history, source references, Clear Chat.
- `requirements.txt`: dependencies.
- `Unlox_Domain_Specific_RAG_Chatbot.ipynb`: core Colab notebook.
- `project_report.pdf`: one-page project report.
- `test_questions.csv`: test question template.

## Notes
- First install/model load may take a few minutes and needs internet access.
- Scanned PDFs may require OCR.
- Semantic similarity alone does not prove a passage answers a question; check evidence and test unanswerable questions.
