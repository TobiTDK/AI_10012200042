# app.py
# Author: [Your Name] | Index: [Your Index Number]
# SourceGround RAG — CS4241 (Academic City) — custom RAG pipeline
# Streamlit: ChatGPT-style chat

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.generation.llm_client import get_llm_provider_label
from src.pipeline.rag_pipeline import RAGPipeline

# (full question, short label for buttons)
SUGGESTED_Q: list[tuple[str, str]] = [
    (
        "How many votes did the NDC receive in the 2020 presidential election?",
        "🗳️ NDC votes (2020 presidential)",
    ),
    (
        "What are the key revenue measures proposed in the 2025 budget?",
        "💰 Key revenue in 2025 budget",
    ),
    (
        "What were the main results in the Greater Accra region?",
        "🗺️ Greater Accra results",
    ),
    (
        "How did the NPP perform in the 2020 presidential compared to the NDC?",
        "⚖️ NPP vs NDC (presidential)",
    ),
    (
        "What fiscal deficit or debt targets are mentioned in the 2025 budget narrative?",
        "📉 Deficit & debt in budget",
    ),
    (
        "Which parties won the most parliamentary seats and in which regions were they strong?",
        "🏛️ Parliamentary results by region",
    ),
]

st.set_page_config(
    page_title="SourceGround",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 50rem; padding-top: 0.75rem; }
    h1#sg-title { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
    p.chat-sub { margin: 0 0 1rem 0; color: #6b7280; font-size: 0.9rem; line-height: 1.4; }
    [data-testid="stChatMessage"] { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar
with st.sidebar:
    st.markdown("### SourceGround")
    st.markdown("**Retrieval**")
    top_k = st.slider("Top-K chunks", 1, 8, 4)
    prompt_version = st.selectbox(
        "Prompt",
        ["v3 (Structured, strict)", "v2 (Hallucination-controlled)", "v1 (Basic)"],
        index=0,
    )
    chunking_method = st.selectbox(
        "Chunking",
        ["fixed (400w, 80w overlap)", "paragraph (300-500w)"],
        index=0,
    )
    st.divider()
    st.markdown("**In replies**")
    show_debug = st.toggle("Debug JSON", value=False)
    show_prompt = st.toggle("Show full prompt", value=False)
    st.divider()
    st.caption("Starters (same as main area)")
    for i, (qtext, short) in enumerate(SUGGESTED_Q[:3]):
        if st.button(short, use_container_width=True, key=f"sb_{i}"):
            st.session_state["pending"] = qtext
            st.rerun()
    st.divider()
    if st.button("➕ New chat", use_container_width=True, type="primary", key="newc"):
        st.session_state["messages"] = []
        for k in list(st.session_state.keys()):
            if k == "pending" or (isinstance(k, str) and k.startswith("pure_ans")):
                st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    try:
        st.caption(f"`{get_llm_provider_label()}`")
    except EnvironmentError:
        st.warning("Add API keys in `.env`", icon="⚠️")
    st.caption("Data: `Ghana_Election_Result.csv` + budget PDF  \nCS4241 — [Your Name] · [Your Index]")

pv_map = {"v3 (Structured, strict)": "v3", "v2 (Hallucination-controlled)": "v2", "v1 (Basic)": "v1"}
cm_map = {"fixed (400w, 80w overlap)": "fixed", "paragraph (300-500w)": "paragraph"}
pv = pv_map[prompt_version]
cm = cm_map[chunking_method]


@st.cache_resource(show_spinner=False)
def get_pipeline(chunking: str) -> RAGPipeline:
    p = RAGPipeline()
    p.initialize(chunking_method=chunking)
    return p


if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, result?, asked?}


# Load pipeline
try:
    with st.status("Loading your knowledge base… (first time ~30s)", expanded=True) as status:
        _pl = get_pipeline(cm)
        status.update(label="Ready", state="complete", expanded=False)
    st.session_state._pl = _pl
except Exception as e:
    st.error(
        "Pipeline initialization failed. On Streamlit Cloud this usually means the `data/` files "
        "are missing from the deployed repo.",
        icon="🚨",
    )
    st.code(str(e))
    st.info(
        "Expected files in `data/`: `Ghana_Election_Result.csv` and "
        "`2025_budget.pdf` (or `2025_Budget_Statement.pdf`)."
    )
    st.stop()

# Title
st.markdown('<h1 id="sg-title">SourceGround</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="chat-sub">RAG on Ghana election data and the 2025 national budget &mdash; answers use your documents.</p>',
    unsafe_allow_html=True,
)

# Suggested questions (like ChatGPT starters) — two rows of chips
def _render_suggestion_chips(key_prefix: str) -> None:
    ra = st.columns(3)
    rb = st.columns(3)
    for i, (qtext, short) in enumerate(SUGGESTED_Q):
        col = ra[i] if i < 3 else rb[i - 3]
        with col:
            if st.button(
                short,
                key=f"{key_prefix}_{i}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["pending"] = qtext
                st.rerun()


st.markdown("**Suggested questions**")
st.caption("Click a button or type your own message at the bottom.")
if not st.session_state.messages:
    _render_suggestion_chips("sug")
else:
    with st.expander("Suggested questions (new topic — same set as above)", expanded=False):
        st.caption("You can also keep chatting below.")
        _render_suggestion_chips("sug_ex")

# Process one queued user message (from chat input, sidebar, or suggestions)
q_in = st.session_state.pop("pending", None)
if q_in and str(q_in).strip():
    u = str(q_in).strip()
    st.session_state.messages.append({"role": "user", "content": u})
    with st.spinner("Retrieving and generating…"):
        res = st.session_state._pl.query(u, top_k=top_k, prompt_version=pv)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": res["response"],
            "result": res,
            "asked": u,
        }
    )
    st.rerun()


def _render_assistant_extras(turn: dict, idx: int) -> None:
    res = turn.get("result")
    if not res:
        return
    if show_prompt:
        with st.expander("Full prompt to the model", expanded=False):
            st.text_area(
                "Prompt",
                value=res.get("final_prompt", ""),
                height=280,
                disabled=True,
                key=f"tp_{idx}",
            )
    with st.expander("Sources & evidence", expanded=False):
        n = len(res.get("scored", []))
        st.caption(f"**Query type:** `{res.get('query_type', '')}` · **Prompt** `{pv}` · **Chunks:** {n}")
        for i, (ch, v_score, b_score, f_score) in enumerate(res.get("scored", []), 1):
            title = f"{i}. {ch.get('chunk_id', '')} — {ch.get('source', '')} · {f_score:.3f}"
            with st.expander(title, expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Vector", f"{v_score:.4f}")
                c2.metric("BM25", f"{b_score:.4f}")
                c3.metric("Combined", f"{f_score:.4f}")
                if ch.get("section_title"):
                    st.caption("Section: " + str(ch["section_title"]))
                if ch.get("keywords"):
                    st.caption("Keywords: " + ", ".join((ch.get("keywords") or [])[:8]))
                t = (ch.get("text") or "")[:1000]
                if len(ch.get("text") or "") > 1000:
                    t += "…"
                st.write(t)
        st.subheader("Score table")
        rows = [
            {
                "Chunk ID": ch.get("chunk_id"),
                "Source": ch.get("source"),
                "Vector": round(v, 4),
                "BM25": round(b, 4),
                "Final": round(f, 4),
            }
            for ch, v, b, f in res.get("scored", [])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if show_debug:
        with st.expander("Debug", expanded=False):
            st.json(
                {
                    "query_type": res.get("query_type"),
                    "prompt_version": pv,
                    "chunking": cm,
                    "top_k": top_k,
                    "selected_chunk_ids": [c.get("chunk_id") for c in (res.get("selected_chunks") or [])],
                }
            )
    uq = turn.get("asked", "")
    if uq:
        with st.expander("Model without your documents (comparison)", expanded=False):
            pkey = f"pure_{idx}"
            if st.button("Get answer (no RAG context)", key=pkey):
                with st.spinner("Calling model…"):
                    st.session_state[f"pure_ans_{idx}"] = st.session_state._pl.query_pure_llm(uq)
            if st.session_state.get(f"pure_ans_{idx}"):
                st.info(st.session_state[f"pure_ans_{idx}"], icon="🤖")
            st.caption("For comparison only; may be wrong on local facts.")


for idx, turn in enumerate(st.session_state.messages):
    with st.chat_message(turn["role"], avatar="🧑" if turn["role"] == "user" else "✨"):
        body = turn.get("content") or ""
        if turn["role"] == "assistant" and body.strip().upper().startswith("[LLM ERROR]"):
            st.error(body)
        else:
            st.markdown(body)
        if turn["role"] == "assistant":
            _render_assistant_extras(turn, idx)

# Chat input (ChatGPT-style fixed at bottom in Streamlit)
if prompt := st.chat_input("Message SourceGround…"):
    st.session_state["pending"] = prompt
    st.rerun()

st.divider()
with st.expander("Run evaluation suite", expanded=False):
    st.caption("Writes `outputs/evaluation_results.json` (may take a few minutes).")
    if st.button("Run evaluation", type="secondary", key="evb"):
        from src.evaluation.run_evaluation import run_evaluation
        with st.spinner("Running…"):
            run_evaluation(st.session_state._pl)
        st.success("Done. See `outputs/evaluation_results.json`.")
