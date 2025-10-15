import os
import json
import time
import argparse
import concurrent.futures
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import requests
from duckduckgo_search import DDGS
import trafilatura
import google.generativeai as genai


#  DATA MODELS
@dataclass
class SourceDocument:
    url: str
    title: Optional[str]
    snippet: Optional[str]
    content: Optional[str]
    fetch_status: str

@dataclass
class PhaseResult:
    name: str
    description: str
    outcome: str
    duration_ms: int

@dataclass
class AgentOutput:
    task: str
    plan: List[str]
    phases: List[PhaseResult]
    evidence_sources: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    response: Dict[str, Any]


# GEMINI CONFIGURATION

API_KEY = "AIzaSyBV3dHAMS83xjEEudW_HaBl49zS4GRTtPo"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

def generate_json_with_gemini(system_instruction: str, prompt: str) -> Dict[str, Any]:
    response = model.generate_content(f"{system_instruction}\n\n{prompt}")
    text = response.text or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON between braces
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first:last + 1])
            except Exception:
                pass
        return {"verdict": "Uncertain", "reasoning": "Failed to parse model output.", "confidence": 0.0}


#  RETRIEVAL

def search_web(query: str, max_results: int = 6, safesearch: str = "moderate") -> List[Dict[str, Any]]:
    results = []
    print(f" Searching DuckDuckGo for: {query}")
    
    # Prioritize credible sources with specific site searches
    credible_sites = [
        "site:nasa.gov", "site:noaa.gov", "site:cdc.gov", "site:who.int", "site:un.org",
        "site:reuters.com", "site:ap.org", "site:bbc.com", "site:cnn.com", "site:npr.org",
        "site:factcheck.org", "site:snopes.com", "site:politifact.com", "site:fullfact.org",
        "site:wikipedia.org", "site:britannica.com", "site:nature.com", "site:sciencedirect.com",
        "site:harvard.edu", "site:mit.edu", "site:stanford.edu", "site:oxford.edu"
    ]
    
    # Try multiple search approaches with credible sources first - PRIORITY ORDER (ENGLISH ONLY)
    search_attempts = [
        # 1. HIGHEST PRIORITY: Government and official sources (English)
        {"query": f"{query} site:gov -site:gov.cn -site:gov.ru -site:gov.in", "safesearch": "moderate", "priority": "government"},
        # 2. Fact-checking organizations (most reliable for verification)
        {"query": f"{query} site:factcheck.org OR site:snopes.com OR site:politifact.com OR site:fullfact.org", "safesearch": "moderate", "priority": "fact_checkers"},
        # 3. Educational institutions (English-speaking countries)
        {"query": f"{query} site:edu -site:edu.cn -site:edu.ru -site:edu.in", "safesearch": "moderate", "priority": "educational"},
        # 4. Scientific organizations (English)
        {"query": f"{query} site:nasa.gov OR site:noaa.gov OR site:cdc.gov OR site:who.int", "safesearch": "moderate", "priority": "scientific"},
        # 5. Major English news organizations
        {"query": f"{query} site:reuters.com OR site:ap.org OR site:bbc.com OR site:npr.org OR site:cnn.com OR site:nytimes.com", "safesearch": "moderate", "priority": "news"},
        # 6. Academic and research sources (English)
        {"query": f"{query} site:nature.com OR site:sciencedirect.com OR site:harvard.edu OR site:mit.edu OR site:stanford.edu OR site:oxford.edu", "safesearch": "moderate", "priority": "academic"},
        # 7. English Wikipedia and Britannica
        {"query": f"{query} site:en.wikipedia.org OR site:britannica.com", "safesearch": "moderate", "priority": "encyclopedia"},
        # 8. General English news search
        {"query": f"{query} news -site:cn -site:ru -site:in", "safesearch": "moderate", "priority": "general_news"},
        # 9. Basic search as last resort
        {"query": query, "safesearch": "moderate", "priority": "fallback"}
    ]
    
    for attempt in search_attempts:
        try:
            priority = attempt.get("priority", "unknown")
            print(f" Trying [{priority}]: {attempt['query']}")
            with DDGS() as ddgs:
                search_results = list(ddgs.text(attempt['query'], safesearch=attempt['safesearch'], max_results=max_results))
                if search_results:
                    # Filter and prioritize credible domains (ENGLISH SOURCES ONLY)
                    credible_results = []
                    for result in search_results:
                        url = result.get("href", "").lower()
                        title = result.get("title", "").lower()
                        
                        # Skip non-English sources
                        if any(skip_domain in url for skip_domain in [
                            ".cn", ".ru", ".in", ".de", ".fr", ".es", ".it", ".jp", ".kr", ".br",
                            "zh.wikipedia", "ru.wikipedia", "de.wikipedia", "fr.wikipedia", "es.wikipedia"
                        ]):
                            continue
                            
                        # Prioritize English credible sources
                        if any(domain in url for domain in [
                            ".gov", ".edu", ".org", "reuters.com", "ap.org", "bbc.com", 
                            "npr.org", "cnn.com", "nytimes.com", "factcheck.org", "snopes.com", 
                            "politifact.com", "fullfact.org", "nasa.gov", "noaa.gov", "cdc.gov", 
                            "who.int", "en.wikipedia.org", "britannica.com", "nature.com", 
                            "harvard.edu", "mit.edu", "stanford.edu", "oxford.edu", "yale.edu",
                            "princeton.edu", "berkeley.edu", "caltech.edu"
                        ]):
                            credible_results.append(result)
                    
                    if credible_results:
                        results.extend(credible_results[:max_results])
                        print(f"  Found {len(credible_results)} credible results from {priority} sources")
                        print(f"  STOPPING SEARCH - Found credible sources!")
                        break
                    else:
                        # Only add non-credible results if we haven't found any credible ones yet
                        if not results:
                            results.extend(search_results[:max_results])
                            print(f"  Found {len(search_results)} results (mixed credibility) - continuing search for better sources")
                        else:
                            print(f"  Found {len(search_results)} results but no credible sources - continuing search")
                else:
                    print(f"  No results for: {attempt['query']}")
        except Exception as e:
            print(f"  Search attempt failed: {e}")
            continue
    
    # If still no results, create credible English fallback results
    if not results:
        print(" All search attempts failed - creating credible English fallback results")
        fallback_results = [
            {"title": f"NASA: {query}", "href": f"https://www.nasa.gov/search/?q={query.replace(' ', '+')}", "body": f"NASA information about {query}"},
            {"title": f"English Wikipedia: {query}", "href": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}", "body": f"English Wikipedia article about {query}"},
            {"title": f"Britannica: {query}", "href": f"https://www.britannica.com/search?query={query.replace(' ', '+')}", "body": f"Britannica encyclopedia entry for {query}"},
            {"title": f"Reuters: {query}", "href": f"https://www.reuters.com/search/news?blob={query.replace(' ', '+')}", "body": f"Reuters news about {query}"},
            {"title": f"FactCheck.org: {query}", "href": f"https://www.factcheck.org/?s={query.replace(' ', '+')}", "body": f"FactCheck.org analysis of {query}"},
            {"title": f"BBC: {query}", "href": f"https://www.bbc.com/search?q={query.replace(' ', '+')}", "body": f"BBC news about {query}"}
        ]
        results = fallback_results
        print(f"Created {len(results)} credible English fallback results")
    
    print(f" Total results: {len(results)}")
    return results

def fetch_url_content(url: str, timeout: int = 15) -> Tuple[str, Optional[str]]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        print(f" Fetching: {url}")
        
        # Try trafilatura first
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded:
            content = trafilatura.extract(downloaded, include_comments=False, favor_recall=True)
            if content and len(content.strip()) > 50:  # Ensure we got meaningful content
                print(f" Trafilatura extracted {len(content)} chars from {url}")
                return ("ok", content)
        
        # Fallback to requests
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        content = trafilatura.extract(resp.text, include_comments=False, favor_recall=True)
        if content and len(content.strip()) > 50:
            print(f" Requests + Trafilatura extracted {len(content)} chars from {url}")
            return ("ok", content)
        else:
            print(f" Insufficient content from {url}")
            return ("error", None)
            
    except Exception as e:
        print(f" Failed to fetch {url}: {e}")
        return ("error", None)

def retrieve_sources(query_or_url: str, mode: str, max_results: int = 6) -> List[SourceDocument]:
    documents: List[SourceDocument] = []
    candidates = []

    #  Normal retrieval
    if mode == "url":
        candidates = [{"title": None, "href": query_or_url, "body": None}]
    else:
        print(f" Searching for: {query_or_url}")
        search_results = search_web(query_or_url, max_results=max_results)
        if not search_results:
            print(" No search results — retrying with expanded keywords...")
            search_results = search_web(query_or_url + " news OR fact check OR site:.org OR site:.gov", max_results=max_results)
        
        # If still no results, try a very basic search
        if not search_results:
            print(" Still no results — trying basic search...")
            search_results = search_web("news " + query_or_url, max_results=max_results)
        
        candidates = [
            {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
            for r in search_results if r.get("href")
        ]
        print(f" Found {len(candidates)} search candidates")
        
        # If we still have no candidates, create some credible English fallback sources
        if not candidates:
            print(" No search candidates found — creating credible English fallback sources...")
            fallback_urls = [
                "https://en.wikipedia.org/wiki/" + query_or_url.replace(" ", "_"),
                "https://www.britannica.com/search?query=" + query_or_url.replace(" ", "+"),
                "https://www.reuters.com/search/news?blob=" + query_or_url.replace(" ", "+"),
                "https://www.factcheck.org/?s=" + query_or_url.replace(" ", "+"),
                "https://www.nasa.gov/search/?q=" + query_or_url.replace(" ", "+"),
                "https://www.cdc.gov/search/?q=" + query_or_url.replace(" ", "+"),
                "https://www.bbc.com/search?q=" + query_or_url.replace(" ", "+")
            ]
            candidates = [{"title": f"Credible English source for {query_or_url}", "href": url, "body": None} for url in fallback_urls]

    #  Parallel fetching
    def _fetch(c: Dict[str, Any]) -> SourceDocument:
        status, content = fetch_url_content(c["href"])
        return SourceDocument(
            url=c["href"],
            title=c.get("title"),
            snippet=c.get("body"),
            content=content,
            fetch_status=status,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        docs = list(executor.map(_fetch, candidates))
        documents.extend(docs)

    successful_sources = [d for d in documents if d.fetch_status == 'ok']
    print(f"📄 Retrieved {len(successful_sources)} working sources")
    
    # Debug: Print source details
    for i, doc in enumerate(successful_sources[:3]):  # Show first 3 sources
        print(f"  Source {i+1}: {doc.url} (title: {doc.title[:50] if doc.title else 'None'}...)")
    
    return documents


#  ANALYSIS PHASES

def plan_phase(task: str) -> Tuple[List[str], PhaseResult]:
    start = time.time()
    plan = [
        "Clarify the claim/topic to verify.",
        "Retrieve recent and reputable sources.",
        "Extract evidence from sources.",
        "Cross-compare for consistency and credibility.",
        "Produce verdict with reasoning and confidence."
    ]
    end = time.time()
    return plan, PhaseResult("Plan", "Define verification strategy and steps.", "plan_ready", int((end - start) * 1000))

def analyze_with_gemini(claim: str, sources: List[SourceDocument]) -> Dict[str, Any]:
    evidence_payload = [
        {
            "url": s.url,
            "title": s.title,
            "snippet": (s.snippet[:300] + "...") if s.snippet and len(s.snippet) > 300 else s.snippet,
            "content_excerpt": (s.content[:1500] + "...") if s.content and len(s.content) > 1500 else s.content,
            "fetch_status": s.fetch_status,
        }
        for s in sources if s.fetch_status == "ok" and (s.content or s.snippet)
    ]

    system_instruction = (
        "You are a fact-checking agent. Return valid JSON with fields: "
        "verdict (True/False/Uncertain), reasoning, confidence (0.0-1.0), "
        "supporting_sources, contradicting_sources."
    )

    user_prompt = f"Task: Verify the claim.\nClaim: {claim}\n\nEvidence: {json.dumps(evidence_payload, ensure_ascii=False)}"
    return generate_json_with_gemini(system_instruction, user_prompt)

def respond_phase(task: str, claim: str, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    verdict = str(analysis_json.get("verdict", "Uncertain"))
    confidence = float(analysis_json.get("confidence", 0.5))
    reasoning = str(analysis_json.get("reasoning", "Insufficient information."))
    supporting = analysis_json.get("supporting_sources", []) or []
    contradicting = analysis_json.get("contradicting_sources", []) or []
    return {
        "task": task,
        "claim": claim,
        "verdict": verdict,
        "confidence": round(max(0.0, min(confidence, 1.0)), 3),
        "reasoning": reasoning,
        "supporting_sources": supporting,
        "contradicting_sources": contradicting,
    }


#  MAIN PIPELINE

def run_agent(claim_or_url: str, mode: str, max_results: int = 6) -> AgentOutput:
    task = "News verification / misinformation detection"

    plan, plan_phase_result = plan_phase(task)

    t0 = time.time()
    sources = retrieve_sources(claim_or_url, mode=mode, max_results=max_results)
    t1 = time.time()
    retrieve_phase_result = PhaseResult("Retrieve", "Search web and extract article text.",
                                        f"retrieved_{len(sources)}_sources", int((t1 - t0) * 1000))

    t2 = time.time()
    analysis_json = analyze_with_gemini(claim_or_url, sources)
    t3 = time.time()
    analyze_phase_result = PhaseResult("Analyze", "Cross-compare evidence and produce structured judgment.",
                                       "analysis_ready", int((t3 - t2) * 1000))

    t4 = time.time()
    response = respond_phase(task, claim_or_url, analysis_json)
    t5 = time.time()
    respond_phase_result = PhaseResult("Respond", "Return structured verdict with confidence and citations.",
                                       "response_ready", int((t5 - t4) * 1000))

    return AgentOutput(task, plan, [plan_phase_result, retrieve_phase_result, analyze_phase_result, respond_phase_result],
                       [asdict(s) for s in sources], analysis_json, response)


#  CLI ENTRY POINT

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--claim", type=str)
    group.add_argument("--url", type=str)
    parser.add_argument("--max_results", type=int, default=6)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    claim_or_url = args.claim or args.url
    mode = "text" if args.claim else "url"

    output = run_agent(claim_or_url, mode, args.max_results)

    result_dict = {
        "agent": "news_verification",
        "reasoning_steps": [
            {"phase": p.name, "outcome": p.outcome, "duration_ms": p.duration_ms, "description": p.description}
            for p in output.phases
        ],
        "plan": output.plan,
        "response": output.response,
        "analysis": output.analysis,
        "evidence_sources": output.evidence_sources,
        "source_count": len([s for s in output.evidence_sources if s.get("fetch_status") == "ok"]),
        "total_sources_attempted": len(output.evidence_sources)
    }

    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

    # Generate organized filename if no save path provided
    if args.save is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a safe filename from the claim/URL
        safe_name = "".join(c for c in claim_or_url[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        # Organize by type: claims vs URLs
        if mode == "text":
            # Create claims directory if it doesn't exist
            os.makedirs("claims", exist_ok=True)
            args.save = f"claims/claim_{timestamp}_{safe_name}.json"
        else:
            # Create urls directory if it doesn't exist
            os.makedirs("urls", exist_ok=True)
            args.save = f"urls/url_{timestamp}_{safe_name}.json"

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"\n Saved output to {args.save}")

if __name__ == "__main__":
    main()
