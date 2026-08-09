import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from ui.backend import DEFAULT_K, EXAMPLE_QUESTIONS, answer_query, backend_status
from ui.styles import inject as inject_styles
from ui.formatting import (
    citation_label,
    dedupe_web_sources,
    format_event_datetime,
    format_score,
    group_sources,
    linkify_citations,
    snippet,
    source_kind,
    source_label,
    split_sources,
    web_link_text,
)

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/account_balance:"

LOG_PATH = REPO_ROOT / "logs" / "ui_turns.jsonl"

st.set_page_config(
    page_title="Boston Public Notices Assistant",
    page_icon=":material/account_balance:",
    layout="centered",
)


def init_state():
    # conversation_id: one per browser session.
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "turn_log" not in st.session_state:
        st.session_state.turn_log = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def log_record(query, answer_text, sources, duration_ms, trace):
    """One line log kept in memory and offered as a download"""
    status = backend_status()
    tools = trace.get("tools_called") or []
    return {
        "conversation_id": st.session_state.conversation_id,
        "attempt": 1,                    
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": "user",
        "source_name": "streamlit_ui",
        "input_message": query,
        "routing_decision": tools or None,
        "target_type": "agent",
        "target_name": "MultiAgent" if status.mode == "live" else "unavailable",
        "output_message": answer_text,
        "duration_ms": duration_ms,
        "agent_latency_ms": trace.get("latency_ms"),
        "source_count": len(sources),
        "guardrail_tripped": trace.get("guardrail_tripped", False),
        "guardrail_reason": trace.get("guardrail_reason"),
    }


def append_log(record):
    """Append one turn to the shared log file.

    Wrapped in try/except on purpose: a read-only directory or a full disk must
    not cost the user their answer. The record is in session state either way,
    so nothing is lost from the transcript download if this fails.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
    except OSError:
        pass


def ask(prompt):
    st.session_state.pending_prompt = prompt


def render_header(status):
    with st.container(key="hero"):
        with st.container(key="eyebrow"):
            st.caption("City of Boston  ·  Official Public Notices")
        st.title("Ask what the city has *actually filed*.")
        st.markdown(
            "Every scheduled meeting and hearing the city has posted, searchable in "
            "plain English. Answers about what is scheduled are written only from "
            "those notices and cite the ones they used, so you can check every claim."
        )

    with st.container(key="herostats"):
        chunks = f"{status.chunks:,}" if status.chunks else "--"
        stats = [
            ("Status", "Live" if status.mode == "live" else "Offline"),
            ("Indexed chunks", chunks),
            ("Coverage", "Jun-Dec 2026"),
        ]
        for column, (label, value) in zip(st.columns(3), stats):
            with column:
                st.metric(label, value, border=True)

    if status.mode != "live":
        st.error(
            f"**The agents did not start** - {status.detail}. Questions cannot be "
            "answered until this is fixed. Check that `chroma_db/` is populated "
            "and `open_ai_api_key.txt` exists in the repo root.",
            icon=":material/error:",
        )

    with st.expander("What this covers", icon=":material/info:"):
        st.markdown(
            "- 150 City of Boston public notices, 2,605 indexed chunks\n"
            "- Event dates from Jun 25 2026 to Dec 16 2026\n"
            "- Current notices are scraped in full, but the **archive is only "
            "~5% sampled** - an older notice may be missing entirely rather than "
            "simply not matching your question\n"
            "- Meeting times are converted to Boston local time\n"
            "- Questions about *when* something is scheduled are answered from the "
            "notices and cited. General questions about how the city works fall back "
            "to a **Boston.gov web search**, which does not yet return citations - "
            "the badge under each answer says which was used\n"
            "- Questions about anything other than Boston city government are refused"
        )


def render_sidebar():
    with st.sidebar, st.container(key="sidenav"):
        st.subheader("Ask about")
        for number, question in enumerate(EXAMPLE_QUESTIONS):
            st.button(
                question,
                key=f"example_{number}",
                width="stretch",
                on_click=ask,
                args=(question,),
            )

        st.subheader("Retrieval")
        st.slider(
            "Sources per answer",
            min_value=2,
            max_value=8,
            value=DEFAULT_K,
            key="k",
            help=(
                "How many notice chunks to retrieve and offer the answer model. "
                "Changing this reloads the agent once, then it is cached."
            ),
        )
        st.toggle("Show retrieval debug", key="show_debug")

        st.subheader("Session")
        st.button(
            "Clear conversation",
            icon=":material/restart_alt:",
            width="stretch",
            on_click=clear_conversation,
        )
        if st.session_state.turn_log:
            st.download_button(
                "Download transcript",
                icon=":material/download:",
                width="stretch",
                data="\n".join(json.dumps(r) for r in st.session_state.turn_log),
                file_name=f"{st.session_state.conversation_id}.jsonl",
                mime="application/x-ndjson",
            )
        st.caption(f"conversation_id `{st.session_state.conversation_id}`")
        if LOG_PATH.exists():
            turns = sum(1 for _ in open(LOG_PATH, encoding="utf-8"))
            st.caption(f"all runs logged to `logs/ui_turns.jsonl` ({turns} turns)")


def clear_conversation():
    # New conversation, new id - the log records are grouped by it.
    st.session_state.turns = []
    st.session_state.turn_log = []
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.suggestions = None
    st.session_state.pending_prompt = None


def render_web_sources(web):
    """Web results: a URL"""
    web = dedupe_web_sources(web)
    if not web:
        return

    st.markdown("**Web results**")
    with st.container(border=True, key=f"web_{id(web)}"):
        st.caption(
            ":material/travel_explore: From a Boston.gov web search, not the "
            "indexed notices. These pages were visited to answer the question."
        )
        for src in web:
            st.markdown(f"- [{web_link_text(src['url'])}]({src['url']})")


def render_sources(sources, turn_index=0):
    notices, web = split_sources(sources)

    if not notices and not web:
        st.caption(
            ":material/search_off: No sources cited. The answer above was not "
            "drawn from the indexed collection - see 'What this covers'."
        )
        return

    if not notices:
        render_web_sources(web)
        return

    st.markdown("**Sources**")

    for _notice_id, items in group_sources(notices):
        head = items[0][1]
        with st.container(border=True, key=f"source_{turn_index}_{_notice_id}"):
            st.markdown(f"**[{getattr(head, 'title', 'No title')}]({head['url']})**")
            when = format_event_datetime(head.get("event_datetime"))
            meta = f":material/event: {when}" if when else ""
            st.caption(f"{meta}  ·  {citation_label([i for i, _ in items])}")
            for index, src in items:
                icon = (
                    ":material/picture_as_pdf:"
                    if src.get("source_type") == "pdf"
                    else ":material/language:"
                )
                with st.expander(f"{index}. {source_kind(src)} - {source_label(src)}", icon=icon):
                    st.markdown(snippet(src.get("text", "")))

    render_web_sources(web)

    if st.session_state.get("show_debug"):
        with st.expander("Retrieval debug", icon=":material/bug_report:"):
            for index, src in enumerate(notices, 1):
                st.markdown(
                    f"`{index}` notice `{src.get('notice_id')}` · "
                    f"{src.get('source_type')} · {format_score(src.get('score', 0.0))}"
                )

TOOL_LABELS = {
    "rag": (":material/folder_open:", "Public notices"),
    "web_search": (":material/travel_explore:", "Boston.gov search"),
}

def render_run_summary(trace):
    if not trace:
        return

    if trace.get("error"):
        st.caption(f":red-badge[:material/error: Agents offline]  ·  {trace['error']}")
        return

    parts = []
    for tool in trace.get("tools_called") or []:
        icon, label = TOOL_LABELS.get(tool, (":material/build:", tool))
        parts.append(f":blue-badge[{icon} {label}]")

    if not parts:
        if trace.get("guardrail_tripped"):
            parts.append(":orange-badge[:material/block: Out of scope]")
        else:
            parts.append(":orange-badge[:material/help: Answered without sources]")

    latency = trace.get("latency_ms")
    if latency:
        parts.append(f"{latency / 1000:.1f}s")

    st.caption("  ·  ".join(parts))

def render_turn(turn, turn_index=0):
    role = turn["role"]
    with st.chat_message(role, avatar=USER_AVATAR if role == "user" else ASSISTANT_AVATAR):
        if role == "user":
            st.markdown(turn["content"])
            return
        st.markdown(linkify_citations(turn["content"], turn["sources"]))
        render_sources(turn["sources"], turn_index)
        render_run_summary(turn.get("trace") or {})

def main():
    init_state()
    inject_styles()
    k = st.session_state.get("k", DEFAULT_K)
    status = backend_status()

    render_header(status)
    render_sidebar()

    for turn_index, turn in enumerate(st.session_state.turns):
        render_turn(turn, turn_index)

    if not st.session_state.turns:
        choice = st.pills(
            "Try asking",
            EXAMPLE_QUESTIONS,
            label_visibility="collapsed",
            key="suggestions",
        )
        if choice:
            ask(choice)

    typed = st.chat_input(
        "Ask about a Boston Public Notice...",
        submit_mode="disable",
    )
    prompt = typed or st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    if not prompt:
        return

    st.session_state.turns.append({"role": "user", "content": prompt})
    render_turn(st.session_state.turns[-1], len(st.session_state.turns) - 1)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        started = time.perf_counter()
        with st.status("Searching public notices...", expanded=False) as box:
            try:
                answer_text, sources, trace = answer_query(prompt, k=k)
                box.update(label="Answer ready", state="complete")
            except Exception as exc:
                answer_text, sources, trace = (
                    f"Something went wrong answering that: {exc}", [], {}
                )
                box.update(label="That did not work", state="error")
        duration_ms = int((time.perf_counter() - started) * 1000)

    st.session_state.turns.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "trace": trace,
        }
    )
    record = log_record(prompt, answer_text, sources, duration_ms, trace)
    st.session_state.turn_log.append(record)
    append_log(record)
    st.rerun()
    
main()