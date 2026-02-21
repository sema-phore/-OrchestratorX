import os
import operator
import re
import json
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import TypedDict, Annotated, Optional, List, Literal, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send

from utility_tools import search_tool, get_stock_price, calculator, rag_tool
from rag_utility import _get_retriever

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Langsmith project name
# ─────────────────────────────────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "OrchestratorX"

# ─────────────────────────────────────────────────────────────
# Shared SQLite connection
# ─────────────────────────────────────────────────────────────
from sqlite_functions import conn

# ─────────────────────────────────────────────────────────────
# LLMs
# ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")
blog_llm = ChatOpenAI(model="gpt-4o-mini")


# ═══════════════════════════════════════════════════════════════
#  CHATBOT GRAPH
# ═══════════════════════════════════════════════════════════════

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

checkpointer = SqliteSaver(conn=conn)

_chat_graph = StateGraph(ChatState)
_chat_graph.add_node("chat_node", chat_node)
_chat_graph.add_node("tools", tool_node)
_chat_graph.add_edge(START, "chat_node")
_chat_graph.add_conditional_edges("chat_node", tools_condition)
_chat_graph.add_edge("tools", "chat_node")

chatBot = _chat_graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    """Retrieve all thread IDs from the database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


# ═══════════════════════════════════════════════════════════════
#  BLOG WRITING AGENT
# ═══════════════════════════════════════════════════════════════

# ── Schemas ──────────────────────────────────────────────────

class BlogTask(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what the reader should understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target word count (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class BlogPlan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[BlogTask]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book", "rag_grounded"]
    queries: List[str] = Field(default_factory=list, description="5-8 specific search queries when needs_research=true")
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class BlogState(TypedDict):
    topic: str
    thread_id: Optional[str]

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    rag_context: List[str]
    plan: Optional[BlogPlan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple], operator.add]   # (task_id, section_md)

    # reducer / images
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str
    progress: str


# ── Router ───────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Your job is to decide which research mode to use. DEFAULT TO HYBRID unless the topic is purely abstract/mathematical with zero dependence on tools, libraries, or recent events.

Modes (choose exactly one):
- closed_book (needs_research=false): ONLY for timeless abstract concepts with NO dependence on real-world tools, APIs, libraries, or events. Example: "Big-O notation explained". Use sparingly.
- hybrid (needs_research=true): DEFAULT choice. Use for ANY topic involving tools, frameworks, models, companies, APIs, best practices, tutorials, comparisons. Example: "How to use LangGraph", "Best vector databases 2024". ALWAYS set needs_research=true here.
- open_book (needs_research=true): Use for news roundups, "this week", "latest", volatile rankings, pricing, policy changes. needs_research=true.
- rag_grounded: Use ONLY when the user has explicitly uploaded a document AND the topic is clearly about that document's content.

IMPORTANT: When in doubt, choose HYBRID with needs_research=true. Almost all interesting blog topics benefit from web research.

If needs_research=true:
- Always output 5–8 specific, scoped search queries.
- Queries must be concrete, e.g. "LangGraph tutorial Python 2024" not just "LangGraph".
- Include at least one query about recent developments or comparisons.
"""


def blog_router_node(state: BlogState) -> dict:
    topic = state["topic"]
    thread_id = state.get("thread_id")
    has_rag = bool(thread_id and _get_retriever(str(thread_id)))

    hint = (
        "\n\nNote: User has uploaded a document. Only use 'rag_grounded' if the topic is clearly about that document's content."
        if has_rag else ""
    )

    decider = blog_llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Blog topic: {topic}\nAs-of date: {state.get('as_of', date.today().isoformat())}{hint}"),
    ])

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    # Safety fallback: if research is needed but no queries generated, create default ones
    queries = decision.queries
    if decision.needs_research and not queries:
        queries = [
            topic,
            f"{topic} tutorial guide",
            f"{topic} best practices 2026",
            f"{topic} examples use cases",
            f"{topic} comparison alternatives",
        ]

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": queries,
        "recency_days": recency_days,
        "progress": f"✓ Routing: {decision.mode} mode | research={decision.needs_research} | queries={len(queries)}",
    }


def blog_route_next(state: BlogState) -> str:
    mode = state.get("mode", "")
    thread_id = state.get("thread_id")

    if mode == "rag_grounded" and thread_id and _get_retriever(str(thread_id)):
        return "rag_retrieval"
    elif state.get("needs_research"):
        return "research"
    else:
        return "orchestrator"


# ── RAG Retrieval ────────────────────────────────────────────

def rag_retrieval_node(state: BlogState) -> dict:
    """Retrieve context from the uploaded PDF to ground blog generation."""
    thread_id = state.get("thread_id")
    topic = state["topic"]

    if not thread_id:
        return {"rag_context": [], "progress": "⚠ No thread ID for RAG"}

    retriever = _get_retriever(str(thread_id))
    if not retriever:
        return {"rag_context": [], "progress": "⚠ No document available"}

    queries = [
        topic,
        f"key concepts related to {topic}",
        f"examples and details about {topic}",
    ]

    all_context = []
    seen = set()
    for q in queries:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_context.append(doc.page_content)

    return {
        "rag_context": all_context[:15],
        "progress": f"✓ Retrieved {len(all_context)} chunks from document",
    }


# ── Research (Tavily) ────────────────────────────────────────

def _tavily_search(query: str, max_results: int = 10) -> List[dict]:
    """Matches friend's exact working implementation."""
    from langchain_community.tools import TavilySearchResults
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized: List[dict] = []
    for r in results or []:
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source": r.get("source"),
        })
    return normalized


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Produce a deduplicated list of EvidenceItem objects from web search results.
Only include items with non-empty URLs. Prefer authoritative sources.
"""


def blog_research_node(state: BlogState) -> dict:
    """Matches friend's exact working implementation — no date filtering that kills results."""
    queries = (state.get("queries") or [])[:10]
    max_results = 10

    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": [], "progress": "⚠ No research results found"}

    extractor = blog_llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ])

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence_list = list(dedup.values())

    return {"evidence": evidence_list, "progress": f"✓ Researched {len(evidence_list)} sources"}


# ── Orchestrator (Plan) ───────────────────────────────────────

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested.
  - If evidence is weak, plan should reflect that (don't invent events).
- rag_grounded: use the document context as primary source. 
  Set requires_research=True where document content is referenced.

Output must match BlogPlan schema.
"""


def blog_orchestrator_node(state: BlogState) -> dict:
    planner = blog_llm.with_structured_output(BlogPlan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    rag_context = state.get("rag_context", [])

    rag_info = ""
    if rag_context:
        rag_info = "\n\nDocument Context (PRIMARY SOURCE):\n" + "\n---\n".join(rag_context[:10])

    evidence_info = ""
    if evidence:
        evidence_info = "\n\nWeb Evidence:\n" + str([e.model_dump() for e in evidence[:12]])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"As-of: {state.get('as_of', date.today().isoformat())} "
            f"(recency_days={state.get('recency_days', 365)})\n"
            f"{'Force blog_kind=news_roundup' if forced_kind else ''}"
            f"{rag_info}"
            f"{evidence_info}"
        )),
    ])

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan, "progress": f"✓ Created outline with {len(plan.tasks)} sections"}


# ── Fanout ────────────────────────────────────────────────────

def blog_fanout(state: BlogState):
    assert state["plan"] is not None
    return [
        Send("blog_worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "as_of": state.get("as_of", date.today().isoformat()),
            "recency_days": state.get("recency_days", 365),
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "rag_context": state.get("rag_context", []),
        })
        for task in state["plan"].tasks
    ]


# ── Worker ────────────────────────────────────────────────────

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", focus on events + implications, not tutorials.

Grounding:
- If mode=="open_book": only introduce claims supported by provided Evidence URLs.
  Attach Markdown links ([Source](URL)) for each supported claim.
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid): cite Evidence URLs for external claims.
- If rag_context provided: use it as primary content source with attribution.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


def blog_worker_node(payload: dict) -> dict:
    task = BlogTask(**payload["task"])
    plan = BlogPlan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    rag_context = payload.get("rag_context", [])

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    rag_text = ""
    if rag_context:
        rag_text = "\n\nDocument Excerpts (PRIMARY):\n" + "\n---\n".join(rag_context[:8])

    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = blog_llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Constraints: {plan.constraints}\n"
            f"Topic: {payload['topic']}\n"
            f"Mode: {payload.get('mode')}\n"
            f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
            f"Section title: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Tags: {task.tags}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n"
            f"{rag_text}\n\n"
            f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
        )),
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}


# ── Reducer subgraph (merge → images) ────────────────────────

def blog_merge_content(state: BlogState) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("blog_merge_content called without plan.")
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md, "progress": "✓ Merged all sections"}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams.
Return strictly GlobalImagePlan.
"""


def blog_decide_images(state: BlogState) -> dict:
    planner = blog_llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog kind: {plan.blog_kind}\n"
            f"Topic: {state['topic']}\n\n"
            "Insert placeholders + propose image prompts.\n\n"
            f"{merged_md}"
        )),
    ])

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
        "progress": f"✓ Planned {len(image_plan.images)} images",
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def blog_generate_and_place_images(state: BlogState) -> dict:
    """Generate images via Gemini and embed as base64 in markdown."""
    import base64

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Metadata footer
    mode_label = state.get("mode", "unknown")
    rag_used = "✓" if state.get("rag_context") else "✗"
    web_used = "✓" if state.get("evidence") else "✗"
    footer = f"\n\n---\n\n*Generated with OrchestratorX | Mode: {mode_label} | Document: {rag_used} | Web Research: {web_used}*"

    if not image_specs:
        return {"final": md + footer, "progress": "✓ Blog complete (no images)"}

    images_dir = BLOGS_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    generated_count = 0
    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
                generated_count += 1
            except Exception as e:
                fallback = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, fallback)
                continue

        try:
            img_bytes = out_path.read_bytes()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            img_md = f"![{spec['alt']}](data:image/png;base64,{b64})\n*{spec.get('caption', '')}*"
            md = md.replace(placeholder, img_md)
        except Exception as e:
            error_block = (
                f"> **[IMAGE ENCODING FAILED]** {spec.get('caption', '')}\n>\n"
                f"> **Error:** {e}\n> Image saved to: `{out_path}`\n"
            )
            md = md.replace(placeholder, error_block)

    return {
        "final": md + footer,
        "progress": f"✓ Generated {generated_count} images, blog complete!",
    }


# ── Blog reducer subgraph ─────────────────────────────────────

_blog_reducer = StateGraph(BlogState)
_blog_reducer.add_node("merge", blog_merge_content)
_blog_reducer.add_node("decide_images", blog_decide_images)
_blog_reducer.add_node("generate_images", blog_generate_and_place_images)
_blog_reducer.add_edge(START, "merge")
_blog_reducer.add_edge("merge", "decide_images")
_blog_reducer.add_edge("decide_images", "generate_images")
_blog_reducer.add_edge("generate_images", END)
blog_reducer = _blog_reducer.compile()

# ── Blog main graph ───────────────────────────────────────────

_blog_graph = StateGraph(BlogState)
_blog_graph.add_node("router", blog_router_node)
_blog_graph.add_node("rag_retrieval", rag_retrieval_node)
_blog_graph.add_node("research", blog_research_node)
_blog_graph.add_node("orchestrator", blog_orchestrator_node)
_blog_graph.add_node("blog_worker", blog_worker_node)
_blog_graph.add_node("reducer", blog_reducer)

_blog_graph.add_edge(START, "router")
_blog_graph.add_conditional_edges("router", blog_route_next, {
    "rag_retrieval": "rag_retrieval",
    "research": "research",
    "orchestrator": "orchestrator",
})
_blog_graph.add_edge("rag_retrieval", "orchestrator")
_blog_graph.add_edge("research", "orchestrator")
_blog_graph.add_conditional_edges("orchestrator", blog_fanout, ["blog_worker"])
_blog_graph.add_edge("blog_worker", "reducer")
_blog_graph.add_edge("reducer", END)

blog_agent = _blog_graph.compile()


# ═══════════════════════════════════════════════════════════════
#  BLOG STORAGE HELPERS
# ═══════════════════════════════════════════════════════════════

BLOGS_DIR = Path("generated_blogs")
BLOGS_DIR.mkdir(exist_ok=True)


def save_blog(blog_content: str, title: str, thread_id: Optional[str] = None) -> dict:
    """Save a generated blog to disk with metadata header."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_title = safe_title.replace(" ", "_")[:50]

    filename = f"{timestamp}_{safe_title}.md"
    filepath = BLOGS_DIR / filename

    metadata = {
        "title": title,
        "generated_at": datetime.now().isoformat(),
        "thread_id": thread_id,
    }
    content_with_meta = f"<!--- {json.dumps(metadata)} --->\n\n{blog_content}"
    filepath.write_text(content_with_meta, encoding="utf-8")

    return {
        "filename": filename,
        "filepath": str(filepath),
        "title": title,
        "timestamp": timestamp,
    }


def list_saved_blogs() -> List[dict]:
    """List all saved blogs, newest first."""
    blogs = []
    for filepath in sorted(BLOGS_DIR.glob("*.md"), reverse=True):
        content = filepath.read_text(encoding="utf-8")
        if content.startswith("<!---"):
            meta_end = content.find("--->\n")
            if meta_end != -1:
                try:
                    meta_json = content[5:meta_end].strip()
                    metadata = json.loads(meta_json)
                    blogs.append({
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "title": metadata.get("title", filepath.stem),
                        "generated_at": metadata.get("generated_at", ""),
                        "thread_id": metadata.get("thread_id"),
                    })
                except Exception:
                    pass
    return blogs


def load_blog(filename: str) -> Optional[str]:
    """Load a saved blog, stripping its metadata header."""
    filepath = BLOGS_DIR / filename
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        if content.startswith("<!---"):
            meta_end = content.find("--->\n")
            if meta_end != -1:
                return content[meta_end + 5:].strip()
        return content
    return None


def delete_blog(filename: str) -> bool:
    """Delete a saved blog file."""
    filepath = BLOGS_DIR / filename
    if filepath.exists():
        filepath.unlink()
        return True
    return False


# ═══════════════════════════════════════════════════════════════
#  BLOG GENERATION RUNNER (called from frontend)
# ═══════════════════════════════════════════════════════════════

def generate_blog(
    topic: str,
    as_of_str: str,
    thread_id: Optional[str] = None,
    progress_callback=None,
) -> dict:
    """
    Stream the blog agent and return the final result dict.

    Returns:
        {
            "content": str,
            "title": str,
            "mode": str,
            "sections_count": int,
            "used_rag": bool,
            "used_web": bool,
            "plan": dict | None,
            "evidence": list,
            "image_specs": list,
        }
    """
    initial_state: Dict[str, Any] = {
        "topic": topic,
        "thread_id": thread_id,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "rag_context": [],
        "plan": None,
        "as_of": as_of_str,
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "progress": "Starting...",
    }

    # Single stream — last emitted state IS the final result (no second invoke)
    result = None
    for event in blog_agent.stream(initial_state, stream_mode="values"):
        result = event
        if progress_callback and isinstance(event, dict) and event.get("progress"):
            progress_callback(event["progress"])

    if result is None:
        result = blog_agent.invoke(initial_state)

    plan_obj = result.get("plan")
    title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else (
        plan_obj.get("blog_title", topic) if isinstance(plan_obj, dict) else topic
    )
    plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj

    # Serialise evidence Pydantic objects
    raw_ev = result.get("evidence") or []
    evidence_dicts = [
        (e.model_dump() if hasattr(e, "model_dump") else e) for e in raw_ev
    ]

    return {
        "content": result.get("final", ""),
        "title": title,
        "mode": result.get("mode", "unknown"),
        "sections_count": len(result.get("sections", [])),
        "used_rag": bool(result.get("rag_context")),
        "used_web": bool(evidence_dicts),
        "plan": plan_dict,
        "evidence": evidence_dicts,
        "image_specs": result.get("image_specs", []),
        "final": result.get("final", ""),
    }