"""
app.py
------
Streamlit chat UI for WealthDesk — Bharat National Bank's AI wealth assistant.
 
Session 13: Streamlit Frontend + Human-in-the-Loop.
 
Architecture:
  - Streamlit chat interface wraps the S12 multi-agent graph (one small change in nodes.py
    adds the optional streaming hook — see _stream_callback below)
  - Every query passes through: Supervisor → [Documents|Rates|Escalate|Decline] → Compliance Agent
  - Human-in-the-Loop (HITL): when the Compliance Agent revises a response,
    the operator sees the revised text and must approve or discard before it
    is shown to the customer.
 
Run:
    streamlit run app.py   (from inside s13/solution/)
 
--- HOW TO ADAPT FOR YOUR LAUNCHPAD PROJECT ---
1. Add `_stream_callback = None` to your nodes.py and call it inside your respond node's
   llm.stream() loop (see wealthdesk/nodes.py for the exact pattern).
2. Replace build_graph() with your own agent's build_graph().
3. Update build_input_state() to match your agent's state keys.
4. Update needs_human_review() to match your own review trigger condition.
5. Update format_route_label() to reflect your agent's routing fields.
6. The HITL and session-state patterns below work as-is.
"""
import sys
import time
from pathlib import Path
from uuid import uuid4
 
import streamlit as st
from dotenv import load_dotenv
 
# --- Path setup ---
# Streamlit runs app.py from the project root, not the package directory.
# This line makes sure Python can find the wealthdesk/ package next to app.py.
# In your launchpad project, change "Path(__file__).parent" to wherever your
# agent package lives.
sys.path.insert(0, str(Path(__file__).parent))
 
# --- Environment variables ---
# Loads GROQ_API_KEY (and anything else) from the .env file in this directory.
# Must happen before importing wealthdesk modules, which read os.getenv() at import time.
load_dotenv()
 
from wealthdesk.agent import build_graph         # noqa: E402
from wealthdesk.config import SAFE_COMPLIANCE_RESPONSE  # noqa: E402
import wealthdesk.nodes as _nodes                # noqa: E402
 
 
# ---------------------------------------------------------------------------
# S13: Token streaming — bridges llm.stream() in nodes.py to the Streamlit UI
# ---------------------------------------------------------------------------
 
class _StreamingState:
    """Receives tokens from nodes.py via _nodes._stream_callback and renders
    them incrementally into a Streamlit placeholder.
 
    HOW IT WORKS:
      nodes.py has a module-level variable `_stream_callback = None`.
      When we set it to an instance of this class, the respond node in nodes.py
      calls it with each token as it arrives from llm.stream().
      This class accumulates the tokens and updates the Streamlit placeholder
      on every call, creating a typewriter effect.
 
    HOW TO USE IN YOUR PROJECT:
      1. Add `_stream_callback = None` to your nodes.py module level.
      2. In your respond node, loop over llm.stream() and call _stream_callback(token).
      3. In app.py, set _nodes._stream_callback = _StreamingState(placeholder) before
         graph.invoke(), then set it back to None after.
    """
 
    def __init__(self, placeholder, token_delay: float = 0.0) -> None:
        # `placeholder` is a Streamlit empty element — st.empty() — that we
        # overwrite on each token. It stays in the same position in the UI.
        self._placeholder = placeholder
        # Accumulates the full response text as tokens arrive.
        self._text = ""
        # Optional per-token sleep in seconds (0 = full speed).
        # Set via the sidebar slider; useful to slow down fast LLMs for demos.
        self._delay = token_delay
 
    def __call__(self, token: str) -> None:
        # Called once per token by nodes.py.
        self._text += token
        # The "▌" cursor at the end signals to the viewer that more is coming.
        # Overwriting the same placeholder avoids flickering — it doesn't add
        # a new element each time.
        self._placeholder.markdown(self._text + "▌")
        if self._delay > 0:
            time.sleep(self._delay)
 
    @property
    def text(self) -> str:
        return self._text
 
 
# ---------------------------------------------------------------------------
# Helper functions (pure, testable -- no Streamlit calls)
# ---------------------------------------------------------------------------
# WHY PURE FUNCTIONS?
# Streamlit calls are not testable with plain pytest. By extracting all logic
# that doesn't need Streamlit into these 5 functions, we can write unit tests
# (see s13/tests/) without running a Streamlit server.
# Rule: if a function calls st.something(), it cannot be in this section.
 
def build_input_state(message: str) -> dict:
    """Return the initial state dict for graph.invoke().
 
    WHY THIS FUNCTION EXISTS:
      graph.invoke() needs the full state schema every time, with all fields
      explicitly reset. Without this reset, stale values from the previous
      turn (e.g. retrieved_docs from an earlier RAG query) leak into the
      new turn's graph execution.
 
    FOR YOUR PROJECT:
      Change these keys to match your agent's WealthDeskState (or equivalent)
      TypedDict in state.py. Every key in your state should appear here,
      reset to its "empty" value.
    """
    return {
        "customer_message":  message,  # the user's input for this turn
        "response":          "",       # filled by the respond node
        "specialist":        "",       # which agent handled the query (rates_agent, etc.)
        "retrieved_docs":    [],       # RAG chunks — must reset each turn to avoid leakage
        "compliance_status": "",       # PASS, REVISED, or FAIL:... — set by compliance node
    }
 
 
def get_thread_config(thread_id: str) -> dict:
    """Return the LangGraph thread config dict.
 
    WHY THIS FUNCTION EXISTS:
      LangGraph requires this exact structure to route checkpointer reads/writes
      to the correct conversation thread. We wrap it in a function so it's
      documented and testable in one place rather than inline everywhere.
 
    FOR YOUR PROJECT:
      This function is identical for any LangGraph agent. Copy it as-is.
      The thread_id comes from st.session_state.thread_id (see _init_session).
    """
    return {"configurable": {"thread_id": thread_id}}
 
 
def compliance_badge(status: str) -> str:
    """Return a short human-readable badge for the compliance status.
 
    WHY startswith("FAIL") INSTEAD OF == "FAIL":
      The compliance node may return "FAIL: guaranteed returns" or
      "FAIL: hallucinated rate" — different violation types in the same FAIL
      family. Using startswith() catches all variants without changing this
      function every time a new FAIL type is added.
 
    FOR YOUR PROJECT:
      If your agent doesn't have a compliance layer, remove this function
      and the badge display in format_route_label().
    """
    if status == "PASS":
        return "✅ Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""  # escalated / declined routes have no compliance status
 
 
def needs_human_review(result: dict) -> bool:
    """Return True when the Compliance Agent revised the response and a human must approve.
 
    WHY THIS FUNCTION EXISTS:
      This single condition is the gate for the HITL form. Extracting it
      makes it testable and gives it a name that explains intent — much
      clearer than an inline `if result.get(...) == "REVISED":` in main().
 
    FOR YOUR PROJECT:
      Change the condition to match your own review trigger. Examples:
        - result.get("confidence_score", 1.0) < 0.7   (low-confidence answer)
        - result.get("escalation_flag", False)         (agent flagged for review)
        - result.get("pii_detected", False)            (PII in response)
    """
    return result.get("compliance_status", "") == "REVISED"
 
 
def format_route_label(result: dict) -> str:
    """Return a one-line route summary for display as a caption below each response.
 
    WHY DISPLAY THIS:
      The route label is a mini audit trail in the UI: which agent answered,
      and was there any compliance action? In production you would log this
      to a database; for the classroom, the caption makes graph routing
      visible without opening LangSmith.
 
    FOR YOUR PROJECT:
      Change "specialist" to whatever field your agent uses to record which
      sub-agent handled the query. Adjust the badge logic to match your
      agent's outcome fields.
    """
    sp    = result.get("specialist", "—")    # e.g. "rates_agent", "documents_agent"
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {sp}"
    if badge:
        label += f" | {badge}"
    return label
 
 
# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
# Functions below call st.* and cannot be unit-tested directly.
# Their logic is kept minimal — they mostly delegate to the pure functions above.
 
def _init_session() -> None:
    """Initialize per-browser-session state on first load.
 
    WHY `if "graph" not in st.session_state`:
      Streamlit re-runs the entire script on every user interaction (button
      click, chat input, slider move). Without this guard, we would create a
      new graph and a new thread_id on every re-run, losing all conversation
      history.
 
    WHY MemorySaver + uuid4:
      MemorySaver keeps conversation history in RAM, scoped to this Python process.
      Each browser tab gets a different uuid4 thread_id, so two tabs don't
      share memory. On page refresh, session_state is cleared and a new
      uuid4 is generated — intentionally starting a fresh conversation.
 
    FOR YOUR PROJECT:
      Replace MemorySaver with SqliteSaver if you need history to survive
      server restarts, or RedisCheckpointSaver for multi-instance deployments.
      In both cases, tie thread_id to the authenticated user's ID rather than
      a random uuid so returning users get their history back.
    """
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        # st.session_state: a dict that persists across re-runs for this browser tab.
        # Assign any value to a key to store it; read it back on the next re-run.
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = str(uuid4())  # unique per browser session
        st.session_state.messages  = []  # list of {"role": "user"/"assistant", "content": "..."}
        st.session_state.routes    = []  # parallel list: one route label per assistant message
 
 
def _sidebar() -> None:
    """Render the sidebar: branding, New Conversation, agent list, demo controls."""
    with st.sidebar:  # st.sidebar: opens a collapsible left panel; everything indented here renders inside it
        st.header("🏦 WealthDesk")       # st.header(): bold H2-size heading inside the sidebar
        st.caption("BNB Customer Assistant")  # st.caption(): small grey text, used for subtitles/metadata
        st.divider()                      # st.divider(): draws a horizontal rule to separate sections
 
        # "New Conversation" clears all session state keys, which causes
        # _init_session() to rebuild the graph and generate a new thread_id
        # on the next re-run. This is the correct way to start fresh —
        # never reuse an existing thread_id for a new conversation because
        # MemorySaver would restore the old message history.
        if st.button("🔄 New Conversation", use_container_width=True):
            # st.button(): renders a clickable button. Returns True only on the re-run
            # triggered by the click — False on every other re-run.
            # use_container_width=True: stretches the button to fill the sidebar width.
            for key in ["graph", "thread_id", "messages", "routes", "pending_hitl"]:
                st.session_state.pop(key, None)
            st.rerun()  # st.rerun(): immediately stops execution and re-runs the script from the top
 
        if "thread_id" in st.session_state:
            # Show first 8 chars of the uuid so the instructor can identify
            # which session they're in without showing the full ID.
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")
 
        st.divider()
        st.subheader("Agents")  # st.subheader(): bold H3-size heading
        st.markdown(             # st.markdown(): renders a markdown string — supports **bold**, *italic*, bullet lists, code blocks
            "- **Supervisor** — classifies query\n"
            "- **Documents Agent** — handles document queries\n"
            "- **Rates Agent** — handles rates & policy queries\n"
            "- **Compliance Agent** — SEBI rules check\n"
            "- **Human-in-the-Loop** — reviews revisions"
        )
 
        st.divider()
        # Token delay slider: slows down streaming so the typewriter cursor
        # effect is visible during demos. 0 = full speed (production default).
        # The value is stored in session_state so the slider position persists
        # across re-runs without being reset to 0 each time.
        st.subheader("Demo settings")
        st.session_state["token_delay"] = st.slider(
            # st.slider(): renders a draggable range widget. Returns the current value on every re-run.
            # Assigning the return value to session_state makes it persist across re-runs.
            "Token delay (ms)",
            min_value=0, max_value=100, value=st.session_state.get("token_delay", 0),
            step=5,
            help="Slow down token streaming so the cursor effect is visible. 0 = full speed.",
            # help=: small tooltip (?) icon shown next to the widget label on hover
        )
 
        st.divider()
        # INSTRUCTOR TOOL: Force HITL Demo
        # The production LLM (gpt-oss-120b) is RLHF-trained to avoid banned
        # phrases like "guaranteed returns" — it won't naturally trigger the
        # HITL form. This button injects a canned compliance-revised response
        # directly into session_state, bypassing the LLM entirely, so the
        # full HITL approval flow can be demonstrated reliably in class.
        # FOR YOUR PROJECT: remove this block from the sidebar. It exists only
        # for classroom demonstration purposes.
        st.subheader("Instructor tools")
        if st.button("⚠️ Force HITL Demo", use_container_width=True,
                     help="Injects a canned compliance-revised response to demonstrate the HITL approval form."):
            # Add the customer message to history so the chat looks natural.
            st.session_state.messages.append({
                "role": "user",
                "content": "Does BNB FD offer guaranteed returns?",
            })
            # Set pending_hitl — this is the HITL gate. When this key exists
            # in session_state, _handle_hitl() shows the approval form and
            # blocks the chat input. The graph is NOT invoked; the response
            # here is the canned revised text that an operator would review.
            st.session_state.pending_hitl = {
                "response": (
                    "BNB Fixed Deposits lock in your interest rate at the time of booking, "
                    "so the rate you agreed to is paid at maturity regardless of market changes. "
                    "Current rates range from 5.5% p.a. (3 months) to 7.3% p.a. (5 years), "
                    "with an additional 0.5% for senior citizens.\n\n"
                    "WealthDesk | Bharat National Bank"
                ),
                "route_label": "Route: rates_agent | ⚠️ Revised",
            }
            st.rerun()
 
 
def _render_history() -> None:
    """Replay all past messages from session_state into the chat UI.
 
    WHY THIS IS NEEDED:
      Streamlit re-runs the entire script on every interaction. The chat
      history is NOT preserved by Streamlit — we preserve it ourselves in
      st.session_state.messages and replay it here on every re-run.
 
    HOW ROUTES ARE MATCHED TO MESSAGES:
      st.session_state.routes is a parallel list to the assistant messages.
      routes[0] is the caption for the first assistant message,
      routes[1] for the second, and so on. assistant_idx tracks which
      assistant message we're currently rendering.
 
    FOR YOUR PROJECT:
      This function works as-is for any agent that stores {"role", "content"}
      dicts in messages and route strings in routes.
    """
    messages = st.session_state.get("messages", [])
    routes   = st.session_state.get("routes",   [])
    assistant_idx = 0
    for msg in messages:
        with st.chat_message(msg["role"]):
            # st.chat_message(role): opens a chat bubble styled for "user" (right-aligned, coloured)
            # or "assistant" (left-aligned, grey). Everything indented here renders inside the bubble.
            st.markdown(msg["content"])  # st.markdown(): renders the message text with markdown formatting
        if msg["role"] == "assistant":
            # Show the route label (e.g. "Route: rates_agent | ✅ Compliant")
            # as a small caption below each assistant bubble.
            if assistant_idx < len(routes):
                st.caption(routes[assistant_idx])  # st.caption(): small grey text rendered below the bubble
            assistant_idx += 1
 
 
def _handle_hitl() -> bool:
    """Render the HITL approval form if a compliance revision is pending.
 
    Returns True while the HITL form is active — main() uses this to
    block the chat input so the operator cannot send a new query until
    they have approved or discarded the pending response.
 
    THE TWO-STATE MODEL:
      "pending_hitl" NOT in session_state → normal chat mode (chat_input visible)
      "pending_hitl" IS in session_state  → HITL mode (approval form visible,
                                             chat_input blocked)
 
    STATE TRANSITIONS:
      Normal → HITL:   set st.session_state.pending_hitl + st.rerun()
      HITL → Normal:   del st.session_state.pending_hitl + st.rerun()
 
    WHY st.rerun():
      Streamlit only re-renders when the script re-runs. After we update
      session_state, we call st.rerun() to immediately trigger a re-run
      so the UI reflects the new state without waiting for user input.
 
    FOR YOUR PROJECT:
      Replace the warning text and form labels with your own review
      workflow. The pattern (set a pending key, check it, show form,
      clear it on decision) works for any operator-review scenario.
    """
    if "pending_hitl" not in st.session_state:
        return False  # nothing pending — normal chat mode
 
    pending = st.session_state.pending_hitl
    # Show a warning banner at the top of the page so the operator knows
    # action is required before the customer can continue.
    st.warning(
        # st.warning(): renders a yellow/amber banner — use for non-critical alerts that need attention.
        # Other variants: st.info() (blue), st.success() (green), st.error() (red).
        "⚠️ **Compliance Review Required** — The Compliance Agent revised this response. "
        "Please review and approve before sending to the customer."
    )
 
    # st.form batches all widgets inside into a single submit event.
    # Without st.form, every keystroke in the text_area would re-run
    # the script and reset the form. The form holds the edit until
    # the operator clicks a submit button.
    with st.form("hitl_approval"):
        # st.form(key): groups widgets so they only trigger a re-run when a submit button is clicked.
        # The key must be unique across the page. All st.* widgets inside are read only on submit.
        edited = st.text_area(
            # st.text_area(): multi-line editable text box. Returns the current text on submit.
            # value=: pre-fills the box. height=: sets the visible height in pixels.
            "Review and edit the response if needed:",
            value=pending["response"],  # pre-filled with the compliance-revised draft
            height=220,
        )
        col1, col2 = st.columns(2)
        # st.columns(n): splits the layout into n equal-width columns.
        # Returns a list of column objects — assign widgets inside each with col.widget().
        approved  = col1.form_submit_button("✅ Approve & Send", use_container_width=True)
        discarded = col2.form_submit_button("❌ Discard",         use_container_width=True)
        # form_submit_button(): a submit button inside st.form. Returns True only on the click re-run.
        # Unlike st.button(), it can only appear inside a st.form block.
 
    if approved:
        # Add the (possibly edited) response to the conversation history
        # and record its route label. Then clear the HITL gate and re-run
        # so the chat returns to normal mode.
        st.session_state.messages.append({"role": "assistant", "content": edited})
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
    elif discarded:
        # Discard: the response never reaches the customer. We show the
        # SAFE_COMPLIANCE_RESPONSE fallback instead — a neutral message
        # that acknowledges the query without making any compliance-risky claim.
        # Without this, the chat would go blank (user message visible, no reply).
        st.session_state.messages.append({
            "role": "assistant",
            "content": SAFE_COMPLIANCE_RESPONSE,
        })
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
 
    return True  # HITL form is active — caller should block chat input
 
 
def main() -> None:
    """Main entry point — called on every Streamlit re-run."""
    st.set_page_config(
        # st.set_page_config(): must be the FIRST Streamlit call in the script.
        # Sets the browser tab title, favicon, and layout mode.
        # layout="wide": uses the full browser width instead of the default narrow centred column.
        page_title="WealthDesk | Bharat National Bank",
        page_icon="🏦",
        layout="wide",
    )
    st.title("🏦 WealthDesk | Bharat National Bank")  # st.title(): large H1 heading at the top of the page
    st.caption("AI-powered wealth assistant — Session 13: Streamlit UI + Human-in-the-Loop")
    # st.caption(): small grey text — used here as a page subtitle
 
    # Step 1: Ensure graph, thread_id, messages, and routes exist in session_state.
    # On the very first load this creates everything. On subsequent re-runs it's a no-op.
    _init_session()
 
    # Step 2: Render the sidebar (branding, controls, demo buttons).
    _sidebar()
 
    # Step 3: Replay all previous messages from session_state into the chat UI.
    # This is necessary because Streamlit re-runs the full script on every event.
    _render_history()
 
    # Step 4: Check whether a HITL review is pending. If so, show the approval
    # form and return True — which means we skip the chat input entirely.
    # The operator must resolve the review before the customer can type again.
    hitl_active = _handle_hitl()
 
    if not hitl_active:
        # Step 5: Show the chat input. st.chat_input() returns None until the
        # user submits a message, at which point Streamlit re-runs with the
        # message in `prompt`.
        prompt = st.chat_input("Ask about BNB deposits, home loans, or investment options…")
        # st.chat_input(): renders a fixed input bar pinned to the bottom of the page.
        # Returns the submitted string on the re-run triggered by the user pressing Enter,
        # None on every other re-run. Only one st.chat_input() is allowed per page.
        if prompt:
            # Immediately add the user message to history and render it.
            # We do this before graph.invoke() so the user sees their message
            # right away, even while the agent is processing.
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):  # st.chat_message("user"): right-aligned bubble with user avatar
                st.markdown(prompt)
 
            # Open an assistant chat bubble early — before we have the response —
            # so the streaming placeholder has somewhere to render into.
            # The placeholder sits inside this bubble and will be overwritten
            # token by token by _StreamingState.__call__().
            with st.chat_message("assistant"):  # st.chat_message("assistant"): left-aligned bubble with bot avatar
                placeholder = st.empty()
                # st.empty(): reserves a slot in the UI with no content yet.
                # Calling placeholder.markdown("text") fills it; placeholder.empty() clears it.
                # This is how we achieve in-place streaming — we overwrite the same slot each token.
 
            # Wire up streaming: set _nodes._stream_callback to our streamer.
            # From this point until we set it back to None, every token emitted
            # by llm.stream() inside nodes.py will call streamer(token).
            # The try/finally ensures the callback is always cleared even if
            # graph.invoke() raises an exception.
            delay_ms = st.session_state.get("token_delay", 0)
            streamer = _StreamingState(placeholder, token_delay=delay_ms / 1000)
            _nodes._stream_callback = streamer
            try:
                # graph.invoke() runs the full agent pipeline:
                #   Supervisor → specialist agent → Compliance Agent
                # build_input_state() resets all state fields to empty values
                # so stale data from the previous turn doesn't leak through.
                # get_thread_config() passes the thread_id so MemorySaver can
                # restore and update the conversation history for this session.
                result = st.session_state.graph.invoke(
                    build_input_state(prompt),
                    config=get_thread_config(st.session_state.thread_id),
                )
            finally:
                # Always clear the callback — even on exception — so the next
                # invocation doesn't accidentally use a stale streamer.
                _nodes._stream_callback = None
 
            # Compute the route label from the graph result before branching.
            route_label = format_route_label(result)
 
            if needs_human_review(result):
                # The Compliance Agent revised the response (compliance_status == "REVISED").
                # Clear the placeholder — we don't want to show the streamed draft,
                # because the operator needs to review and possibly edit it first.
                placeholder.empty()
                # Store the revised response and its route label so _handle_hitl()
                # can pre-fill the approval form on the next re-run.
                st.session_state.pending_hitl = {
                    "response":    result["response"],
                    "route_label": route_label,
                }
                # Trigger a re-run immediately so the HITL form appears without
                # the user having to do anything.
                st.rerun()
            else:
                # Normal path: response is compliant (PASS) or escalated/declined
                # (no compliance check). Replace the streaming placeholder with
                # the final clean response (removes the ▌ cursor).
                placeholder.markdown(result["response"])
                # placeholder.markdown(): overwrites the st.empty() slot with the final text.
                # At this point streaming is done — this call removes the ▌ cursor.
                st.caption(route_label)
                # st.caption(): renders the route label (e.g. "Route: rates_agent | ✅ Compliant")
                # as small grey text directly below the assistant bubble.
                # Save to session_state so _render_history() can replay these
                # on the next re-run.
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                st.session_state.routes.append(route_label)
 
 
if __name__ == "__main__":
    main()