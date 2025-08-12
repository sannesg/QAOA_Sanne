import streamlit as st
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # (optional, disables the warning)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qiskit.visualization.circuit.matplotlib")

from planner import Planner

CODE_FENCE_RE = re.compile(r'```(?P<lang>[^\n]*)\n(?P<code>.*?)```', re.DOTALL)

def escape_leading_hashes(text: str) -> str:
    """Escape leading '#' on lines so they don't render as markdown headers.
       Only escapes the first '#' on each line that starts with it.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # if line starts with '#' after stripping left whitespace, escape it
        if stripped.startswith("#"):
            # preserve leading indentation, escape the first '#'
            prefix_len = len(line) - len(stripped)
            lines[i] = line[:prefix_len] + "\\" + line[prefix_len:]
    return "\n".join(lines)

def render_message_content(content: str):
    """Render a message that may contain fenced code blocks and markdown."""
    last = 0
    for m in CODE_FENCE_RE.finditer(content):
        pre = content[last:m.start()]
        if pre.strip():
            st.markdown(escape_leading_hashes(pre))
        lang = m.group("lang").strip() or None
        code = m.group("code").rstrip("\n")
        # use st.code for code blocks (syntax highlighting supported)
        st.code(code, language=lang)
        last = m.end()
    # remaining tail
    tail = content[last:]
    if tail.strip():
        st.markdown(escape_leading_hashes(tail))

@st.cache_resource
def get_planner():
    return Planner()

planner = get_planner()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_input" not in st.session_state:
    st.session_state.current_input = ""
    
if "pending_response" not in st.session_state:
    st.session_state.pending_response = False
    
st.title("QAOA Agent Chat")

if "send_clicked" not in st.session_state:
    st.session_state.send_clicked = False

if st.session_state.send_clicked:
    query = st.session_state.current_input
    if query.strip():
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.current_input = ""  # Safe: before widget is rendered
        st.session_state.pending_response = True
    st.session_state.send_clicked = False
    st.rerun()

if st.session_state.pending_response and st.session_state.messages:
    
    with st.spinner("Thinking..."):
        result = planner(st.session_state.messages[-1]["content"])
    # st.session_state.messages.append({"role": "assistant", "content": result, "media": captured_media})
    # st.session_state.pending_response = False
    
    captured_media = []
    
    if "```python" in result:
        code = re.search(r"```python\n(.*?)\n```", result, re.DOTALL)
        if code:
            code_str = code.group(1)
            code_str = re.sub(r"plt\.show\(\)", "", code_str)  # Remove plt.show() calls
            exec_globals = globals()
            exec_locals = {}
            
            try:
                matplotlib.use("Agg", force=True)
                
                exec(code_str, exec_globals, exec_locals)
                seen_figs = set()
                seen_pil = set()

                # 1. Active pyplot figure(s)
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    if id(fig) not in seen_figs:
                        captured_media.append(("matplotlib", fig))
                        seen_figs.add(id(fig))

                # 2. Figures / images from exec_locals
                for value in exec_locals.values():
                    if hasattr(value, "savefig") and hasattr(value, "add_subplot"):  # Matplotlib Figure
                        if id(value) not in seen_figs:
                            captured_media.append(("matplotlib", value))
                            seen_figs.add(id(value))
                    elif isinstance(value, Image.Image):  # PIL Image
                        if id(value) not in seen_pil:
                            captured_media.append(("pil", value))
                            seen_pil.add(id(value))
                            
            except Exception:
                pass

    st.session_state.last_media = captured_media

    st.session_state.messages.append({
        "role": "assistant",
        "content": result,
        "media": captured_media
    })

    st.session_state.pending_response = False

# Chat history display
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         render_message_content(msg["content"])

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        render_message_content(msg["content"])
        
        # If this is the last assistant message, attach captured media
        if msg["role"] == "assistant" and idx == len(st.session_state.messages) - 1:
            if "last_media" in st.session_state:
                for kind, obj in st.session_state.last_media:
                    if kind == "matplotlib":
                        st.pyplot(obj)
                        plt.close(obj)
                    elif kind in ("pil", "bytes"):
                        st.image(obj)

# User input
query = st.text_area(
    "Your message:",
    height=150,
    placeholder="Ask me about QAOA...",
    value= st.session_state.current_input,
    key="current_input",
    label_visibility="collapsed"  # Hide the label visually
)

if st.button("Send"):
    st.session_state.send_clicked = True
    st.rerun()