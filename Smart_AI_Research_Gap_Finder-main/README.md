# Smart Research Gap Finder

An AI research-analysis application built with Streamlit. It combines **Advanced RAG, Agentic RAG and a multi-agent workflow** to analyze uploaded research papers and produce evidence-backed potential research opportunities.

## What it does
- Multi-PDF upload and page-aware text extraction
- Structure-aware chunking with source/page metadata
- FAISS vector retrieval with MMR diversity and lightweight lexical reranking
- Paper analysis, literature comparison, critical review and trend analysis
- Potential research-gap detection with evidence references
- Verification agent that checks whether proposed gaps are supported by the uploaded literature
- Research idea and proposal generation
- RAG-powered research assistant chat

> The system identifies **potential** research gaps. A generated opportunity is not a guarantee of novelty; researchers should verify recent literature before making novelty claims.

## Architecture
```text
Streamlit UI
    ↓
Research Manager
    ↓
Advanced RAG Engine
    ├─ PDF parser
    ├─ structure-aware chunks
    ├─ MiniLM embeddings
    ├─ FAISS
    ├─ MMR retrieval
    └─ lexical reranking
    ↓
Multi-Agent Workflow
    ├─ Paper Analysis
    ├─ Comparison
    ├─ Critical Review
    ├─ Trend Analysis
    ├─ Gap Detection
    ├─ Verification
    ├─ Research Ideas
    └─ Proposal
    ↓
Results + Research Assistant
```

## Local setup
Python 3.11+ is recommended.

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Smart_AI_Research_Gap_Finder-main
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
streamlit run app.py
```

## Streamlit Community Cloud deployment
1. Push this folder to GitHub.
2. In Streamlit Community Cloud, create a new app and select the GitHub repository.
3. Set the main file to `app.py`.
4. Deploy.
5. Open the app settings and add the secret:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

6. Save and reboot/redeploy the app.

**Never commit `.streamlit/secrets.toml` or a `.env` file.** The included `.streamlit/secrets.toml.example` is only a template.

## Important deployment notes
- The embedding model downloads on first startup, so the first launch can take longer.
- FAISS is built in memory for each Streamlit session; uploaded PDFs are not permanently stored by this app.
- Groq usage depends on the account/model limits available to your API key.
- For large paper collections, reduce the number/size of PDFs or tune `RAG_TOP_K` and context settings.

## Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | Groq authentication |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | LLM model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_TOP_K` | `8` | Retrieval depth |
| `MAX_CONTEXT_CHARS` | `18000` | Maximum LLM context used per call |

## Security
API keys are read from environment variables/Streamlit Secrets. Do not hardcode credentials or upload secrets to GitHub.

## License
Educational and research use.
