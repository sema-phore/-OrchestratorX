"""
⚙️ OrchestratorX
=================
Unified application combining:
  • 💬 Multi-Utility Chatbot  — conversational AI with web search, stock prices,
                                calculator, and per-thread RAG over uploaded PDFs.
  • ✍️ Blog Writing Agent     — LangGraph pipeline that researches, plans, writes,
                                and illustrates long-form technical blog posts.

Run with:
    streamlit run streamlit_frontend.py
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ─── Chatbot stack ────────────────────────────────────────────────────────────
from langgraph_backend import (
    chatBot,
    blog_agent,
    generate_blog,
    save_blog,
    list_saved_blogs,
    load_blog,
    delete_blog,
    retrieve_all_threads,
    BLOGS_DIR,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sqlite_functions import save_thread_metadata
from rag_utility import (
    ingest_pdf,
    load_existing_retriever,
    get_thread_metadata as get_pdf_metadata,
    _get_retriever,
)
from utility_tools import set_rag_thread_id
from streamlit_utility_functions import (
    generate_thread_id,
    reset_chat,
    add_thread,
    load_conversation,
    generate_conversation_name,
    load_threads_from_database,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Page config  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚙️ OrchestratorX",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def filter_messages(messages: list) -> list[dict]:
    """Convert LangGraph BaseMessages to plain UI dicts, dropping tool noise."""
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, HumanMessage):
            role, content = "user", msg.content
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None) and not msg.content:
                continue
            role, content = "assistant", msg.content
        else:
            continue
        if content and content.strip():
            result.append({"role": role, "content": content})
    return result


# ─── Blog writer helpers ──────────────────────────────────────────────────────

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress using stream_mode='values'.
    Each yielded step is the FULL state after that node completes.
    The last step IS the final state — no second invoke() needed.
    Falls back to a direct invoke if streaming fails.
    """
    try:
        last_state = None
        for step in graph_app.stream(inputs, stream_mode="values"):
            last_state = step
            yield ("values", step)
        if last_state is not None:
            yield ("final", last_state)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        current.update(payload)
    return current


# ── Markdown renderer that supports base64 and local images ──────────────────
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")

# Mojibake / encoding-artifact patterns produced when UTF-8 bytes are
# mis-decoded as latin-1 or Windows-1252, plus stray Unicode replacement chars.
_ENCODING_FIXES: List[Tuple[re.Pattern, str]] = [
    # Common multi-byte mojibake sequences → correct UTF-8 characters
    (re.compile(r'â€™'), "'"),
    (re.compile(r'â€œ'), '"'),
    (re.compile(r'â€\x9d'), '"'),
    (re.compile(r'â€"'), '—'),
    (re.compile(r'â€"'), '–'),
    (re.compile(r'â€¦'), '…'),
    (re.compile(r'â€¢'), '•'),
    (re.compile(r'Ã©'), 'é'),
    (re.compile(r'Ã¨'), 'è'),
    (re.compile(r'Ã '), 'à'),
    (re.compile(r'Ã¢'), 'â'),
    (re.compile(r'Ã®'), 'î'),
    (re.compile(r'Ã´'), 'ô'),
    (re.compile(r'Ã»'), 'û'),
    (re.compile(r'Ã§'), 'ç'),
    # Numeric reference-style artifacts like â96, â97, ◆96, ◆97 etc.
    (re.compile(r'[◆â]\d{2,3}'), ''),
    # Stray citation superscripts embedded mid-sentence e.g. "text\x9696 more"
    (re.compile(r'[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]\d{1,3}'), ''),
    # Lone diamond / box question marks sometimes followed by digits
    (re.compile(r'[□◇]{1}\d*'), ''),
    # Unicode replacement character (U+FFFD) and lookalikes
    (re.compile(r'[\ufffd\x96\x97\x92\x93\x94]'), ''),
    # Stray lone â, Ã at word boundaries followed by nothing useful
    (re.compile(r'\bâ\b|\bÃ\b'), ''),
]


def _clean_encoding(text: str) -> str:
    """
    Fix common mojibake and stray encoding artifacts in LLM-generated markdown.
    Strategy:
      1. Try to re-encode as latin-1 then decode as utf-8 (fixes classic mojibake).
      2. Apply pattern-based replacements for residual artifacts.
      3. Strip the Unicode replacement character.
    """
    # Step 1 — attempt round-trip fix for classic latin-1 → utf-8 mojibake
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        fixed = text

    # Step 2 — apply regex replacements
    for pattern, replacement in _ENCODING_FIXES:
        fixed = pattern.sub(replacement, fixed)

    # Step 3 — strip any remaining replacement chars
    fixed = fixed.replace("\ufffd", "")

    return fixed


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_images(md: str):
    """
    Render a markdown string, fixing encoding artifacts and handling local/remote
    images inline. Wraps content in a readable max-width container.
    """
    # Fix encoding first
    md = _clean_encoding(md)

    matches = list(_MD_IMG_RE.finditer(md))

    # Wrap in a readable container
    st.markdown(
        "<style>.blog-preview{max-width:860px; margin:auto; line-height:1.75; "
        "font-size:1.02rem;} .blog-preview h1,.blog-preview h2,.blog-preview h3"
        "{margin-top:1.6em; margin-bottom:0.4em;} .blog-preview p{margin-bottom:1em;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="blog-preview">', unsafe_allow_html=True)

    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last: m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None

        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("data:") or src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")

        i += 1

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session-state bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

# ── App-level ─────────────────────────────────────────────────────────────────
if "active_tool" not in st.session_state:
    st.session_state["active_tool"] = "chatbot"   # "chatbot" | "blog_agent"

# ── Chatbot ───────────────────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state["initialized"] = False

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if not st.session_state["initialized"]:
    st.session_state["chat_threads"] = load_threads_from_database()
    st.session_state["initialized"] = True

if "thread_id" not in st.session_state:
    if st.session_state["chat_threads"]:
        st.session_state["thread_id"] = st.session_state["chat_threads"][0]["id"]
    else:
        st.session_state["thread_id"] = generate_thread_id()
        add_thread(st.session_state["thread_id"])

if "message_history" not in st.session_state:
    raw = load_conversation(st.session_state["thread_id"])
    st.session_state["message_history"] = filter_messages(raw) if raw else []

if "pdf_loaded" not in st.session_state:
    st.session_state["pdf_loaded"] = False

retriever = load_existing_retriever(st.session_state["thread_id"])
if retriever:
    st.session_state["pdf_loaded"] = True

# ── Blog Agent ────────────────────────────────────────────────────────────────
if "last_blog_out" not in st.session_state:
    st.session_state["last_blog_out"] = None

if "blog_logs" not in st.session_state:
    st.session_state["blog_logs"] = []


# ═══════════════════════════════════════════════════════════════════════════════
# ════════════════════════════  SIDEBAR  ═══════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Brand header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding: 12px 0 6px 0;'>
            <span style='font-size:2rem;'>⚙️</span><br/>
            <span style='font-size:1.4rem; font-weight:700; letter-spacing:0.04em;'>OrchestratorX</span><br/>
            <span style='font-size:0.78rem; color:#888;'>Your AI command centre</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Tool switcher ─────────────────────────────────────────────────────────
    st.markdown("**🔀 Switch Tool**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "💬 Chatbot",
            use_container_width=True,
            type="primary" if st.session_state["active_tool"] == "chatbot" else "secondary",
        ):
            st.session_state["active_tool"] = "chatbot"
            st.rerun()
    with col_b:
        if st.button(
            "✍️ Blog Agent",
            use_container_width=True,
            type="primary" if st.session_state["active_tool"] == "blog_agent" else "secondary",
        ):
            st.session_state["active_tool"] = "blog_agent"
            st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # CHATBOT sidebar section
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state["active_tool"] == "chatbot":

        st.markdown("#### 💬 Conversations")

        if st.button("➕ New Chat", use_container_width=True):
            reset_chat()
            st.session_state["pdf_loaded"] = False
            st.rerun()

        # Thread list
        for thread in st.session_state["chat_threads"][::-1]:
            tid, tname = thread["id"], thread["name"]
            is_active = tid == st.session_state["thread_id"]
            label = f"{'▶ ' if is_active else ''}{tname}"
            if st.button(label, key=f"thread_{tid}", use_container_width=True):
                st.session_state["thread_id"] = tid
                raw = load_conversation(tid)
                st.session_state["message_history"] = filter_messages(raw) if raw else []
                retriever = load_existing_retriever(tid)
                st.session_state["pdf_loaded"] = retriever is not None
                st.rerun()

        st.divider()

        # PDF upload
        st.markdown("#### 📄 Document Upload")
        current_pdf_meta = get_pdf_metadata(st.session_state["thread_id"])
        if current_pdf_meta:
            st.success(f"✅ {current_pdf_meta.get('filename', 'Document')}")
            st.caption(
                f"📄 {current_pdf_meta.get('documents', 'N/A')} pages · "
                f"✂️ {current_pdf_meta.get('chunks', 'N/A')} chunks"
            )
        else:
            st.info("No document uploaded for this chat")

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
        if uploaded_file and not current_pdf_meta:
            with st.status("Processing document…", expanded=True) as doc_status:
                st.write("📖 Reading PDF…")
                file_bytes = uploaded_file.read()
                st.write("✂️ Splitting into chunks…")
                result = ingest_pdf(
                    file_bytes=file_bytes,
                    thread_id=st.session_state["thread_id"],
                    filename=uploaded_file.name,
                )
                if result.get("success", False):
                    st.write("✅ Processing complete!")
                    doc_status.update(label="✅ Document ready!", state="complete")
                    st.session_state["pdf_loaded"] = True
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message', 'Unknown error')}")
                    doc_status.update(label="❌ Failed", state="error")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOG AGENT sidebar section
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("#### ✍️ Blog Writing Agent")

        blog_topic = st.text_area(
            "Topic",
            height=120,
            placeholder="e.g. 'How RAG systems work', 'Latest AI weekly roundup'…",
            key="blog_topic_input",
        )
        blog_as_of = st.date_input("As-of date", value=date.today(), key="blog_as_of")

        use_thread_rag = st.checkbox(
            "🔗 Use current chat's document",
            value=False,
            help="Ground the blog in the PDF uploaded in your active chat thread.",
            key="blog_use_rag",
        )

        run_blog_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

        st.divider()

        # ── Saved blogs list ──────────────────────────────────────────────────
        st.markdown("#### 📚 Saved Blogs")
        saved_blogs = list_saved_blogs()
        if not saved_blogs:
            st.caption("No saved blogs yet.")
            selected_label = None
        else:
            options = [f"{b['title']} ({b['generated_at'][:10]})" for b in saved_blogs[:30]]
            file_by_label = {
                f"{b['title']} ({b['generated_at'][:10]})": b["filename"]
                for b in saved_blogs[:30]
            }
            selected_label = st.radio(
                "Select a blog to load",
                options=options,
                index=0,
                label_visibility="collapsed",
                key="saved_blog_radio",
            )

            col_load, col_del = st.columns(2)
            if col_load.button("📂 Load", use_container_width=True):
                filename = file_by_label.get(selected_label)
                if filename:
                    content = load_blog(filename)
                    blog_meta = next((b for b in saved_blogs if b["filename"] == filename), {})
                    if content:
                        st.session_state["last_blog_out"] = {
                            "final": content,
                            "title": blog_meta.get("title", "Blog"),
                            "plan": None,
                            "evidence": [],
                            "image_specs": [],
                            "mode": "loaded",
                        }
                        st.session_state["active_tool"] = "blog_agent"
                        st.rerun()

            if col_del.button("🗑️ Delete", use_container_width=True):
                filename = file_by_label.get(selected_label)
                if filename and delete_blog(filename):
                    st.success("Deleted.")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════  MAIN AREA  ═══════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
#  CHATBOT VIEW
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state["active_tool"] == "chatbot":

    st.title("💬 Multi-Utility Chatbot")
    st.caption("Web search · Stock prices · Calculator · Document Q&A")

    # PDF status banner
    if _get_retriever(st.session_state["thread_id"]):
        pdf_meta = get_pdf_metadata(st.session_state["thread_id"])
        if pdf_meta:
            st.info(
                f"💡 **Document loaded:** {pdf_meta.get('filename', 'your document')} "
                "— Ask anything about it!"
            )

    # Render conversation
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type here…")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        set_rag_thread_id(st.session_state["thread_id"])

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_turn",
        }

        with st.chat_message("assistant"):
            status_holder = {"box": None}
            response_parts: List[str] = []

            def ai_only_stream():
                for message_chunk, metadata in chatBot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        label = (
                            "🔍 Searching document…"
                            if tool_name == "rag_tool"
                            else f"🔧 Using `{tool_name}`…"
                        )
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(label, expanded=True)
                        else:
                            status_holder["box"].update(label=label, state="running", expanded=True)
                        continue

                    if isinstance(message_chunk, AIMessage):
                        if getattr(message_chunk, "tool_calls", None) and not message_chunk.content:
                            continue
                        if message_chunk.content:
                            response_parts.append(message_chunk.content)
                            yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Done", state="complete", expanded=False)

        final_message = "".join(response_parts) if response_parts else ai_message
        if final_message and isinstance(final_message, str) and final_message.strip():
            st.session_state["message_history"].append({"role": "assistant", "content": final_message})

        # Auto-name conversation after ≥ 3 messages
        current_thread = next(
            (t for t in st.session_state["chat_threads"] if t["id"] == st.session_state["thread_id"]),
            None,
        )
        if (
            current_thread
            and not current_thread.get("named", False)
            and len(st.session_state["message_history"]) >= 3
        ):
            try:
                name = generate_conversation_name(
                    st.session_state["message_history"],
                    st.session_state["thread_id"],
                )
                current_thread["name"] = name
                current_thread["named"] = True
                save_thread_metadata(st.session_state["thread_id"], name, True)
            except Exception as e:
                print(f"[Naming] {e}")

        # Strip any empty messages
        st.session_state["message_history"] = [
            m for m in st.session_state["message_history"] if m.get("content", "").strip()
        ]

        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  BLOG AGENT VIEW
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.title("✍️ Blog Writing Agent")
    st.caption("Research · Plan · Write · Illustrate — powered by LangGraph")

    tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
        ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"]
    )

    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)

    # ── Run the agent if the button was pressed ───────────────────────────────
    if "run_blog_btn" in dir() and run_blog_btn:
        topic_val = st.session_state.get("blog_topic_input", "").strip()
        as_of_val = st.session_state.get("blog_as_of", date.today())

        if not topic_val:
            st.warning("Please enter a topic in the sidebar.")
            st.stop()

        thread_id_for_rag = st.session_state["thread_id"] if use_thread_rag else None

        inputs: Dict[str, Any] = {
            "topic": topic_val,
            "thread_id": thread_id_for_rag,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "rag_context": [],
            "plan": None,
            "as_of": as_of_val.isoformat() if hasattr(as_of_val, "isoformat") else str(as_of_val),
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
            "progress": "Starting...",
        }

        status = st.status("Running Blog Writing Agent…", expanded=True)
        progress_area = st.empty()
        current_state: Dict[str, Any] = {}
        last_node = None

        for kind, payload in try_stream(blog_agent, inputs):
            if kind == "values":
                current_state = extract_latest_state(current_state, payload)

                node_progress = current_state.get("progress", "")
                if node_progress and node_progress != last_node:
                    status.write(f"➡️ {node_progress}")
                    last_node = node_progress

                raw_evidence = current_state.get("evidence", []) or []
                evidence_count = len(raw_evidence)

                summary = {
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "queries": (current_state.get("queries", []) or [])[:5],
                    "evidence_count": evidence_count,
                    "tasks": len((current_state.get("plan") or {}).get("tasks", []))
                    if isinstance(current_state.get("plan"), dict)
                    else (len(current_state["plan"].tasks) if current_state.get("plan") else None),
                    "images_planned": len(current_state.get("image_specs", []) or []),
                    "sections_done": len(current_state.get("sections", []) or []),
                }
                progress_area.json(summary)
                log(f"[values] mode={current_state.get('mode')} evidence={evidence_count} progress={node_progress}")

            elif kind == "final":
                out = payload

                # Serialise Pydantic objects so session_state stays JSON-safe
                plan_obj = out.get("plan")
                plan_dict = (
                    plan_obj.model_dump()
                    if hasattr(plan_obj, "model_dump")
                    else plan_obj
                )
                out["plan"] = plan_dict

                raw_ev = out.get("evidence") or []
                out["evidence"] = [
                    (e.model_dump() if hasattr(e, "model_dump") else e)
                    for e in raw_ev
                ]

                st.session_state["last_blog_out"] = out
                status.update(label="✅ Blog generated!", state="complete", expanded=False)
                log(f"[final] mode={out.get('mode')} evidence={len(out['evidence'])} final_len={len(out.get('final',''))}")

                # Auto-save
                if out.get("final"):
                    title = (plan_dict or {}).get("blog_title", topic_val) if plan_dict else topic_val
                    save_blog(out["final"], title, thread_id_for_rag)

        st.session_state["blog_logs"].extend(logs)
        st.rerun()

    # ── Render last result ────────────────────────────────────────────────────
    out = st.session_state.get("last_blog_out")

    if out:
        # ── Plan tab ──────────────────────────────────────────────────────────
        with tab_plan:
            st.subheader("Blog Plan")
            plan_dict = out.get("plan")
            if not plan_dict:
                st.info("No plan available (blog may have been loaded from disk).")
            else:
                if not isinstance(plan_dict, dict):
                    plan_dict = json.loads(json.dumps(plan_dict, default=str))

                st.write("**Title:**", plan_dict.get("blog_title"))
                cols = st.columns(3)
                cols[0].write("**Audience:** " + str(plan_dict.get("audience", "—")))
                cols[1].write("**Tone:** " + str(plan_dict.get("tone", "—")))
                cols[2].write("**Type:** " + str(plan_dict.get("blog_kind", "—")))

                mode_badge = out.get("mode", "")
                if mode_badge:
                    st.caption(f"Generation mode: `{mode_badge}`")

                tasks = plan_dict.get("tasks", [])
                if tasks:
                    df = pd.DataFrame([
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]).sort_values("id")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    with st.expander("Full task details (JSON)"):
                        st.json(tasks)

        # ── Evidence tab ──────────────────────────────────────────────────────
        with tab_evidence:
            st.subheader("Research Evidence")
            evidence = out.get("evidence") or []
            if not evidence:
                st.info("No evidence collected — blog generated in closed-book or rag-grounded mode.")
            else:
                rows = []
                for e in evidence:
                    if hasattr(e, "model_dump"):
                        e = e.model_dump()
                    rows.append({
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Preview tab ───────────────────────────────────────────────────────
        with tab_preview:
            st.subheader("Blog Preview")
            final_md = out.get("final") or ""
            if not final_md:
                st.warning("No markdown generated yet.")
            else:
                render_markdown_with_images(final_md)

                plan_dict = out.get("plan") or {}
                blog_title = (
                    plan_dict.get("blog_title", "")
                    if isinstance(plan_dict, dict)
                    else ""
                ) or out.get("title", "blog")

                md_filename = f"{safe_slug(blog_title)}.md"

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇️ Download Markdown",
                        data=final_md.encode("utf-8"),
                        file_name=md_filename,
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with dl_col2:
                    bundle = bundle_zip(final_md, md_filename, BLOGS_DIR / "images")
                    st.download_button(
                        "📦 Download Bundle (MD + images)",
                        data=bundle,
                        file_name=f"{safe_slug(blog_title)}_bundle.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

        # ── Images tab ────────────────────────────────────────────────────────
        with tab_images:
            st.subheader("Generated Images")
            specs = out.get("image_specs") or []
            images_dir = BLOGS_DIR / "images"

            if not specs and not images_dir.exists():
                st.info("No images generated for this blog.")
            else:
                if specs:
                    st.write("**Image specifications:**")
                    st.json(specs)

                if images_dir.exists():
                    files = [p for p in images_dir.iterdir() if p.is_file()]
                    if files:
                        for p in sorted(files):
                            st.image(str(p), caption=p.name, use_container_width=True)
                    else:
                        st.info("images/ directory is empty.")

        # ── Logs tab ──────────────────────────────────────────────────────────
        with tab_logs:
            st.subheader("Agent Logs")
            all_logs = st.session_state.get("blog_logs", [])
            st.text_area(
                "Event log",
                value="\n\n".join(all_logs[-80:]),
                height=520,
            )
            if st.button("🧹 Clear logs"):
                st.session_state["blog_logs"] = []
                st.rerun()

    else:
        with tab_preview:
            st.info("Enter a topic in the sidebar and click **🚀 Generate Blog** to begin.")
        with tab_plan:
            st.info("The blog plan will appear here after generation.")
        with tab_evidence:
            st.info("Research evidence will appear here for hybrid/open-book topics.")
        with tab_images:
            st.info("Generated images will appear here.")
        with tab_logs:
            st.info("Agent execution logs will appear here.")
