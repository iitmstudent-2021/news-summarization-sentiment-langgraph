# AI Agent for News Summarization & Sentiment Analysis

> **Course:** IIT Madras BS in Data Science & Applications — DSAI Lab  
> **Term:** January 2026 | Week 8 Bonus Assignment  
> **Framework:** LangGraph + LangChain + Groq (Free LLM API)  
> **Model:** `llama-3.3-70b-versatile` via Groq

---

## What This Project Does

This project builds an **AI Agent** that can:

1. **Fetch** a news article from a URL (or accept pre-provided text)
2. **Summarize** the article in 3–5 sentences using an LLM
3. **Analyse the sentiment** (Positive / Negative / Neutral) using a rule-based tool
4. **Return a structured JSON response** with summary, sentiment label, confidence score, and a news snippet

---

## Project Structure

```
Week-8_Bonus_Assgn/
│
├── news_agent_notebook.ipynb   ← Main Jupyter Notebook (step-by-step, submit this)
├── V_1.ipynb                   ← Same notebook run on Google Colab (with saved outputs)
├── news_agent.py               ← Standalone Python script version of the agent
├── requirements.txt            ← All Python dependencies
└── README.md                   ← This file
```

> **PS:** `news_agent.py` and `news_agent_notebook.ipynb` contain the same logic.  
> The notebook is better for demonstration and submission because it shows outputs cell by cell.  
> `V_1.ipynb` is the Colab-executed version with all real outputs already saved.

---

## Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Agent Framework | `langgraph` | Controls the multi-step agent workflow |
| LLM Integration | `langchain-groq` | Connects to Groq's free LLM API |
| LLM Model | `llama-3.3-70b-versatile` | Summarizes news articles |
| Web Scraping | `requests` + `beautifulsoup4` | Fetches and parses news from URLs |
| Tools | `langchain_core.tools` | Defines `fetch_news` and `analyze_sentiment` |
| State Management | `TypedDict` + `Annotated` | Tracks agent state across nodes |

> **PS:** Groq is completely **free** — sign up at [console.groq.com](https://console.groq.com) and get an API key instantly. No credit card required.

---

## Architecture — LangGraph Flow

```
                    ┌────────────────────────────────┐
                    │       entry_router              │
                    │  (Conditional Entry Point)      │
                    │                                 │
                    │  text provided? → summarize     │
                    │  URL provided?  → agent         │
                    └────────┬───────────────┬────────┘
                             │               │
                          [agent]       [summarize] ◄──────────┐
                             │               │                  │
                  tool_calls?│        [sentiment]               │
                    yes ▼  no┤               │                  │
                         [tools:     [build_response]           │
                        fetch_news]        │                    │
                             │            END                   │
                   [process_results] ─────────────────────────►─┘
```

### How the flow works:

1. **entry_router** checks if news text is already provided
   - If YES → skip fetching, jump straight to `summarize`
   - If NO (URL given) → go to `agent`

2. **agent** — the LLM decides whether to call the `fetch_news` tool
   - If it calls a tool → go to `tools`
   - If not → go to `summarize`

3. **tools** → executes `fetch_news`, scrapes the URL

4. **process_results** → extracts the scraped text from the ToolMessage and saves it to state

5. **summarize** → LLM reads the article and writes a 3–5 sentence summary

6. **sentiment** → calls `analyze_sentiment` tool directly, returns Positive/Negative/Neutral + score

7. **build_response** → packages everything into a clean JSON output

> **PS:** LangGraph works by passing a shared **state dictionary** between nodes. Each node reads from the state, does its job, and returns only the fields it updated (partial dict). This is why each node returns `{"key": value}` instead of the whole state.

---

## Agent State

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # chat messages (auto-appends)
    news_text: str        # raw article text (fetched or pre-provided)
    summary: str          # LLM-generated summary
    sentiment: str        # Positive / Negative / Neutral
    sentiment_score: float  # confidence score 0.0 to 1.0
    final_response: dict  # final structured output
```

> **PS:** `messages` uses `Annotated[list, operator.add]` — this tells LangGraph to **append** new messages instead of replacing them. Without this, multiple nodes trying to update `messages` would cause an `InvalidUpdateError`.

---

## Tool 1 — `fetch_news`

```python
@tool
def fetch_news(url: str) -> str:
    ...
```

**What it does:**
- Takes a news article URL
- Sends an HTTP GET request with a browser-like User-Agent header
- Parses the HTML with BeautifulSoup
- Removes junk tags: `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, `<form>`
- Tries to extract `<article>` tag first; falls back to all `<p>` tags
- Cleans up extra whitespace
- Returns first 8000 characters (to stay within LLM token limits)

**Returns:** Clean article text string, or an error message if fetch fails.

> **PS:** We cap at 8000 characters because LLMs have a token limit. Sending the full webpage (which can be 50,000+ characters) would exceed the context window and fail or cost more tokens.

> **PS:** The `User-Agent` header is set to mimic a real browser. Many news websites block requests that don't have a browser-like User-Agent, returning a 403 Forbidden error.

---

## Tool 2 — `analyze_sentiment`

```python
@tool
def analyze_sentiment(text: str) -> dict:
    ...
```

**What it does:**
- Takes article text as input
- Counts how many **positive keywords** appear (e.g., growth, record, excellent, milestone, soar)
- Counts how many **negative keywords** appear (e.g., crisis, fraud, collapse, layoff, recession)
- Computes ratio of each over total word count
- Returns a label and a confidence score

**Scoring logic:**
```
pos_count > neg_count  →  Positive,  score = 0.5 + (pos_ratio × 10), capped at 0.99
neg_count > pos_count  →  Negative,  score = 0.5 + (neg_ratio × 10), capped at 0.99
neither found          →  Neutral,   score = 0.50
equal counts           →  Neutral,   score = 0.50
```

**Returns:** `{"label": "Positive", "score": 0.99}`

> **PS:** This is a **rule-based** (lexicon-based) approach — no ML model is needed. It is fast, free, and deterministic. A transformer-based approach (e.g., HuggingFace `distilbert-sentiment`) would be more accurate but requires more setup and compute.

> **PS:** We call this tool **directly** in the `sentiment_node` using `analyze_sentiment.invoke({"text": text})` instead of asking the LLM to call it. This is more reliable — we don't need the LLM to decide to call a rule-based function.

---

## Routing Functions

### `entry_router`
```python
def entry_router(state: AgentState) -> str:
    return "summarize" if state.get("news_text", "").strip() else "agent"
```
Checks if `news_text` is already filled in the state. If yes, skip fetching. If no, let the agent handle it.

> **PS:** This avoids a bug where the LLM would try to call `fetch_news` even when text was already provided — wasting an API call and causing errors.

### `after_agent`
```python
def after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "summarize"
```
Checks if the last message from the LLM contains tool calls. If yes, route to the tools node to execute them.

---

## Graph Nodes

### `agent_node`
- Calls the LLM (with `fetch_news` tool bound)
- LLM reads the user message and decides whether to call `fetch_news`
- Returns `{"messages": [response]}`

### `process_fetch_results`
- Scans messages in reverse to find the latest `ToolMessage`
- Extracts the fetched article text and stores it in `news_text`
- Returns `{"news_text": "..."}`

### `summarize_node`
- Sends a prompt to the LLM: *"You are a professional news editor. Summarize in 3–5 sentences..."*
- Returns `{"summary": "..."}`

### `sentiment_node`
- Calls `analyze_sentiment.invoke()` directly on the article text
- Returns `{"sentiment": "Positive", "sentiment_score": 0.99}`

### `build_response_node`
- Combines summary + sentiment + news snippet into a clean dict
- Returns `{"final_response": {...}}`

---

## Final Output Format

```json
{
  "summary": "Apple Inc. reported record quarterly earnings, exceeding Wall Street expectations with a 15% revenue increase driven by strong iPhone sales and Services growth...",
  "sentiment": {
    "label": "Positive",
    "score": 0.99
  },
  "news_snippet": "Apple Inc. reported record-breaking quarterly earnings on Thursday, surpassing Wall Street expectations..."
}
```

---

## Two Input Modes

### Mode 1 — Pre-provided text
```python
make_state(text="Your news article text here...")
```
- Skips `fetch_news` tool entirely
- Goes directly to `summarize` → `sentiment` → `build_response`

### Mode 2 — URL fetching
```python
make_state(url="https://en.wikipedia.org/wiki/Artificial_intelligence")
```
- LLM agent calls `fetch_news` tool
- Scrapes article text from the URL
- Then proceeds to summarize and sentiment

---

## How to Run

### Option A — Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `news_agent_notebook.ipynb`
3. In Cell 2, set your Groq API key:
   ```python
   os.environ["GROQ_API_KEY"] = "gsk_your_key_here"
   ```
4. Click **Runtime → Run All**

### Option B — Local Terminal
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (Windows)
set GROQ_API_KEY=gsk_your_key_here

# Set API key (Mac/Linux)
export GROQ_API_KEY=gsk_your_key_here

# Run
python news_agent.py
```

> **PS:** Get your free Groq API key at [console.groq.com](https://console.groq.com) → API Keys → Create Key. It looks like `gsk_...`. No credit card needed.

---

## Demo Results

### Demo 1 — Positive News (Apple earnings)
```json
{
  "summary": "Apple Inc. reported record quarterly earnings, exceeding Wall Street expectations with a 15% revenue increase...",
  "sentiment": { "label": "Positive", "score": 0.99 },
  "news_snippet": "Apple Inc. reported record-breaking quarterly earnings..."
}
```

### Demo 2 — Negative News (Banking scandal)
```json
{
  "summary": "A major banking scandal has been uncovered, revealing widespread fraud and sparking a sharp decline in investor confidence...",
  "sentiment": { "label": "Negative", "score": 0.99 },
  "news_snippet": "A major banking scandal has rocked financial markets..."
}
```

### Demo 3 — URL Fetch (Wikipedia AI article)
```json
{
  "summary": "Artificial intelligence (AI) refers to computational systems that perform tasks associated with human intelligence...",
  "sentiment": { "label": "Negative", "score": 0.58 },
  "news_snippet": "Artificial intelligence (AI) is the capability of computational systems..."
}
```

> **PS:** The Wikipedia AI article scores as slightly Negative (0.58) because it contains words like "risk", "danger", "threat", "conflict" when discussing AI safety concerns — this is expected behaviour from the rule-based analyser.

---

## Dependencies

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-groq>=0.1.0
langchain-community>=0.3.0
beautifulsoup4>=4.12.0
requests>=2.31.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Used Groq instead of OpenAI/Anthropic | Groq is completely free with generous rate limits |
| Rule-based sentiment (not ML model) | No GPU/model download needed, fast and reliable |
| `Annotated[list, operator.add]` for messages | Prevents LangGraph `InvalidUpdateError` when multiple nodes update messages |
| Nodes return partial dicts | Correct LangGraph pattern — only return keys you updated |
| `set_conditional_entry_point` | Allows skipping the fetch step when text is pre-provided |
| Text capped at 8000 chars for fetch | Stays within LLM context window limits |
| Direct tool call for sentiment | More reliable than asking LLM to call a rule-based function |
