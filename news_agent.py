"""
AI Agent for News Summarization & Sentiment Analysis
Built with LangGraph + Groq (Free LLM API - llama-3.3-70b-versatile)
"""

import os
import re
import json
import operator
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ─────────────────────────────────────────────
# 1. Agent State
#    messages uses operator.add so multiple nodes
#    can safely append without conflict
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # auto-appends across nodes
    news_text: str
    summary: str
    sentiment: str
    sentiment_score: float
    final_response: dict


# ─────────────────────────────────────────────
# 2. Tools
# ─────────────────────────────────────────────

@tool
def fetch_news(url: str) -> str:
    """
    Fetch the main text content of a news article from the given URL.
    Returns the extracted article text or an error message.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)

        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 100:
            return "Could not extract sufficient article text from URL."
        return text[:8000]
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


@tool
def analyze_sentiment(text: str) -> dict:
    """
    Perform rule-based sentiment analysis on the provided text.
    Returns a dict with 'label' (Positive / Negative / Neutral) and 'score' (0.0-1.0).
    """
    positive_words = {
        "good", "great", "excellent", "positive", "success", "win", "won",
        "achievement", "improve", "improvement", "growth", "benefit", "gain",
        "rise", "increase", "strong", "better", "best", "happy", "joy",
        "profit", "record", "breakthrough", "innovative", "leading", "advance",
        "recover", "recovery", "boom", "surge", "thriving", "optimistic",
        "upbeat", "promising", "milestone", "progress", "robust", "soar",
    }
    negative_words = {
        "bad", "terrible", "negative", "fail", "failure", "loss", "lost",
        "decline", "drop", "fall", "crisis", "problem", "issue", "concern",
        "risk", "threat", "danger", "worse", "worst", "sad", "tragic",
        "disaster", "collapse", "crash", "fraud", "scandal", "violence",
        "war", "attack", "injury", "dead", "death", "recession", "layoff",
        "bankrupt", "controversy", "protest", "conflict", "weak", "plunge",
    }

    words = re.findall(r"\b\w+\b", text.lower())
    total = len(words) if words else 1
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    if pos_count == 0 and neg_count == 0:
        return {"label": "Neutral", "score": 0.50}
    elif pos_count > neg_count:
        return {"label": "Positive", "score": round(0.5 + min(pos_count / total * 10, 0.49), 2)}
    elif neg_count > pos_count:
        return {"label": "Negative", "score": round(0.5 + min(neg_count / total * 10, 0.49), 2)}
    else:
        return {"label": "Neutral", "score": 0.50}


# ─────────────────────────────────────────────
# 3. Graph Nodes
# ─────────────────────────────────────────────

def agent_node(state: AgentState, llm_with_tools) -> dict:
    """LLM decides whether to call fetch_news tool (for URL input)."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def process_fetch_results(state: AgentState) -> dict:
    """Extract fetched article text from the last ToolMessage."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            return {"news_text": msg.content}
    return {}


def summarize_node(state: AgentState, llm) -> dict:
    """Ask the LLM to summarize the news text."""
    text = state.get("news_text", "")
    if not text:
        return {"summary": "No news text available to summarize."}
    prompt = (
        "You are a professional news editor. "
        "Summarize the following news article in 3-5 concise sentences, "
        "capturing the key facts and main message.\n\n"
        f"Article:\n{text}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content.strip()}


def sentiment_node(state: AgentState) -> dict:
    """Call analyze_sentiment tool directly — rule-based, fast, free."""
    text = state.get("news_text", "")
    if not text:
        return {"sentiment": "Unknown", "sentiment_score": 0.0}
    result = analyze_sentiment.invoke({"text": text[:3000]})
    return {"sentiment": result["label"], "sentiment_score": result["score"]}


def build_response_node(state: AgentState) -> dict:
    """Assemble the final structured response."""
    snippet = state.get("news_text", "")
    return {
        "final_response": {
            "summary": state.get("summary", "N/A"),
            "sentiment": {
                "label": state.get("sentiment", "Unknown"),
                "score": state.get("sentiment_score", 0.0),
            },
            "news_snippet": snippet[:300] + "..." if len(snippet) > 300 else snippet,
        }
    }


# ─────────────────────────────────────────────
# 4. Routing
# ─────────────────────────────────────────────

def entry_router(state: AgentState) -> str:
    """Skip fetch if text already provided, else let agent decide."""
    return "summarize" if state.get("news_text", "").strip() else "agent"


def after_agent(state: AgentState) -> str:
    """Did the agent request a tool call?"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "summarize"


# ─────────────────────────────────────────────
# 5. Graph Builder
# ─────────────────────────────────────────────

def build_graph():
    llm_base       = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm_with_tools = llm_base.bind_tools([fetch_news])
    tool_node      = ToolNode([fetch_news])

    graph = StateGraph(AgentState)

    graph.add_node("agent",           lambda s: agent_node(s, llm_with_tools))
    graph.add_node("tools",           tool_node)
    graph.add_node("process_results", process_fetch_results)
    graph.add_node("summarize",       lambda s: summarize_node(s, llm_base))
    graph.add_node("sentiment",       sentiment_node)
    graph.add_node("build_response",  build_response_node)

    graph.set_conditional_entry_point(
        entry_router,
        {"agent": "agent", "summarize": "summarize"},
    )
    graph.add_conditional_edges("agent", after_agent,
        {"tools": "tools", "summarize": "summarize"})
    graph.add_edge("tools",           "process_results")
    graph.add_edge("process_results", "summarize")
    graph.add_edge("summarize",       "sentiment")
    graph.add_edge("sentiment",       "build_response")
    graph.add_edge("build_response",  END)

    return graph.compile()


# ─────────────────────────────────────────────
# 6. Helper
# ─────────────────────────────────────────────

def make_state(text: str = "", url: str = "") -> AgentState:
    if text:
        return {
            "messages": [HumanMessage(content="Analyse the provided news article.")],
            "news_text": text,
            "summary": "", "sentiment": "", "sentiment_score": 0.0, "final_response": {},
        }
    return {
        "messages": [
            HumanMessage(content=f"Fetch the news article from this URL and analyse it: {url}\nUse the fetch_news tool.")
        ],
        "news_text": "", "summary": "", "sentiment": "", "sentiment_score": 0.0, "final_response": {},
    }


# ─────────────────────────────────────────────
# 7. CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AI News Summarization & Sentiment Analysis Agent")
    print("  Powered by: LangGraph + Groq (llama-3.3-70b-versatile)")
    print("=" * 60)

    app = build_graph()

    # Demo 1: Positive news
    positive_text = (
        "Apple Inc. reported record-breaking quarterly earnings on Thursday, surpassing "
        "Wall Street expectations with a 15% increase in revenue driven by strong iPhone "
        "sales and Services growth. CEO Tim Cook called it a milestone quarter, citing "
        "robust demand across all product categories including the new Apple Vision Pro. "
        "The company's stock soared 5% in after-hours trading following the announcement, "
        "with analysts upgrading their 12-month price targets."
    )
    print("\n[Demo 1] Positive News")
    print("-" * 40)
    r1 = app.invoke(make_state(text=positive_text))
    print(json.dumps(r1["final_response"], indent=2))

    # Demo 2: Negative news
    negative_text = (
        "A major banking scandal has rocked financial markets after regulators uncovered "
        "widespread fraud at one of the country's largest financial institutions. "
        "The crisis led to a sharp decline in investor confidence, with stock markets "
        "plunging more than 8% in a single session. Thousands of employees face layoffs "
        "as the bank teeters on the edge of collapse. The disaster triggered protests "
        "outside headquarters. Analysts warn the fallout could tip the economy into recession."
    )
    print("\n[Demo 2] Negative News")
    print("-" * 40)
    r2 = app.invoke(make_state(text=negative_text))
    print(json.dumps(r2["final_response"], indent=2))

    # Demo 3: URL fetch
    print("\n[Demo 3] Fetch from URL")
    print("-" * 40)
    r3 = app.invoke(make_state(url="https://en.wikipedia.org/wiki/Artificial_intelligence"))
    print(json.dumps(r3["final_response"], indent=2))
