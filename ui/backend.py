import asyncio
from typing import Literal, NamedTuple

import streamlit as st

DEFAULT_K = 4

UNAVAILABLE_TEXT = (
    "The agents are not running, so there is nothing to answer from. "
    "The banner at the top of the page says why."
)


class BackendStatus(NamedTuple):
    mode: Literal["live", "error"]
    detail: str                
    chunks: int = 0             


@st.cache_resource(show_spinner="Starting the agents (~30s on first run)...")
def _load_agent():
    try:
        from orchestrator_agent import MultiAgent
        from rag_agent.rag_agent import get_agent

        count = get_agent().chunk_count()
        if count == 0:
            return None, BackendStatus("error", "chroma_db is empty - no notices indexed")

        return MultiAgent(), BackendStatus("live", f"MultiAgent, {count} chunks", count)

    except ImportError as exc:
        return None, BackendStatus("error", f"agents not importable ({exc})")
    except Exception as exc:
        return None, BackendStatus("error", f"agents failed to start ({type(exc).__name__}: {exc})")


def backend_status() -> BackendStatus:
    return _load_agent()[1]


def answer_query(query: str, k: int = DEFAULT_K):
    agent, status = _load_agent()
    if agent is None:
        return UNAVAILABLE_TEXT, [], {"error": status.detail}

    from rag_agent.rag_agent import get_agent
    get_agent().k = k

    return asyncio.run(agent.answer(query))

EXAMPLE_QUESTIONS = [
    "When is the retirement board meeting?",
    "When is the next meeting?",
    "What is happening in August 2026?",
    "Which meetings allow public testimony?",
    "Have any public notices been cancelled?",
    "What are the wine regions of France?",
]