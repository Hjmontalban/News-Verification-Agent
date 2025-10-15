## News Verification Agent (Gemini)

An autonomous misinformation-detection agent using the Gemini API. It follows a Plan → Retrieve → Analyze → Respond reasoning sequence, performs real web retrieval, and returns structured JSON with reasoning steps, confidence, and citations.

### Agent type and rationale
- **Type**: Retrieval-augmented, multi-phase fact-checking agent.
- **Why**: News and social claims often require fresh, cross-source evidence. A small, deterministic pipeline with explicit phases keeps behavior transparent and auditable while leveraging an LLM only for the judgment step.

### Why this agent
- **News verification** is a practical use-case for agentic reasoning and retrieval.
- Emphasizes cross-source evidence and calibrated confidence.

### System architecture
- **Planner**: Outlines the verification strategy.
- **Retriever**: Uses DuckDuckGo with English-only credible source prioritization (government, fact-checkers, news orgs, scientific sources) and Trafilatura + Requests to extract article text.
- **Analyzer (Gemini)**: Consumes claim + evidence and outputs a structured verdict (JSON).
- **Responder**: Formats final response with verdict, confidence, and cited sources.

```
User Claim/URL → Plan → Retrieve (DDG + trafilatura) → Analyze (Gemini) → Respond (JSON)
```

### Reasoning process
1. Clarify claim/topic.
2. Retrieve from credible sources (government, fact-checkers, news orgs, scientific sources).
3. Extract evidence from sources with robust fallback mechanisms.
4. Cross-compare evidence; prefer high-credibility outlets.
5. Produce verdict with reasoning and confidence; cite sources.

### Tools and libraries
- `google-generativeai` (Gemini API)
- `duckduckgo_search` (search with credible source prioritization)
- `trafilatura` (web content extraction)
- `requests`

### Credible source prioritization (English Only)
The agent prioritizes English-language sources in this order:
1. **Government & Official**: NASA, NOAA, CDC, WHO, UN, all .gov domains (excluding non-English)
2. **Fact-Checking Organizations**: FactCheck.org, Snopes, PolitiFact, FullFact
3. **Educational Institutions**: Harvard, MIT, Stanford, Oxford, Yale, Princeton, Berkeley, Caltech
4. **Scientific Organizations**: NASA, NOAA, CDC, WHO, Nature.com, ScienceDirect
5. **Major English News**: Reuters, Associated Press, BBC, NPR, CNN, New York Times
6. **Academic Sources**: Nature.com, ScienceDirect, top universities
7. **English Wikipedia & Britannica**: en.wikipedia.org, britannica.com
8. **Fallback Sources**: If searches fail, creates credible English fallback URLs

**Language Filtering**: Automatically excludes non-English domains (.cn, .ru, .in, .de, .fr, .es, .it, .jp, .kr, .br) and non-English Wikipedia versions.

### ⚠️ Security Warning
**This project contains a hardcoded API key for demonstration purposes only.**
- The Gemini API key is hardcoded in `agent.py` (line 43: `API_KEY = "..."`)
- **DO NOT use this in production** - it's insecure and violates security best practices
- **For production**: Use environment variables or secure secret management
- **For local development**: Consider replacing with `os.environ["GEMINI_API_KEY"]`

### Setup
1. Python 3.10+
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. **⚠️ API Key Configuration**:
   - **WARNING**: The current code uses a **hardcoded API key** in `agent.py` (variable `API_KEY`). 
   - For local testing it will work as-is, but this is **insecure for production use**.
   - **Recommended**: Replace with `os.environ["GEMINI_API_KEY"]` or use a secure secret manager.
   - **Security Note**: Never commit API keys to version control in real projects.

Optional (recommended) virtual environment on Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Usage
Verify a textual claim:
```bash
python agent.py --claim "The WHO declared XYZ on DATE"
```

Assess an article URL:
```bash
python agent.py --url "https://example.com/article" --save report.json
```

CLI flags:
- `--claim` or `--url`: exactly one is required
- `--max_results`: number of web results to gather (default 6)
- `--save`: path to write a full JSON report (optional - if not provided, auto-organizes into `claims/` or `urls/` folders)

Model configuration:
- The code currently uses `gemini-2.0-flash` via `google-generativeai` and does not expose a `--model` flag.

Quick start (Windows PowerShell):
```powershell
# (optional) create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# verify a claim
python agent.py --claim "The WHO declared XYZ on DATE"

# verify a URL and save output
python agent.py --url "https://example.com/article" --save agent_output.json
```

### Output
The agent prints structured JSON with:
- `reasoning_steps` (each phase with duration)
- `plan`
- `response` (verdict, confidence, reasoning, citations)
- `analysis` (Gemini’s structured judgment)
- `evidence_sources` (raw sources and fetch status)

### Notes
- Confidence is calibrated by source agreement and credibility; unclear or conflicting evidence yields `Uncertain` verdicts.
- Be mindful of rate limits and robots policies when retrieving content.

### Current implementation details (as of this repo)
- API key is set in `agent.py` as `API_KEY`. For production, replace with `os.environ["GEMINI_API_KEY"]` or a secure secret manager.
- Model is fixed to `gemini-2.0-flash`.
- Output is automatically organized: claims save to `claims/claim_YYYYMMDD_HHMMSS_[name].json`, URLs save to `urls/url_YYYYMMDD_HHMMSS_[name].json`.
- **English-only source prioritization**: Searches government sites (.gov excluding non-English), educational institutions (.edu excluding non-English), fact-checkers (FactCheck.org, Snopes, PolitiFact), major English news orgs (Reuters, AP, BBC, NPR, CNN, NYT), and scientific sources (NASA, NOAA, Nature) first.
- **Language filtering**: Automatically excludes non-English domains and Wikipedia versions to ensure all evidence sources are in English.
- **Robust retrieval**: Multiple search strategies with fallback mechanisms ensure sources are found even when primary searches fail.
- **Enhanced debugging**: Detailed logging shows search attempts, source credibility, language filtering, and extraction success rates.


