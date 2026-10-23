import base64
import io
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from ollama import Client
import time

# Optional packages used only by the relevant features:
# pip install diffusers transformers accelerate safetensors torch
# pip install pypdf pandas pillow

APP_TITLE = "MARVELLOUS AI"
TEXT_MODEL_DEFAULT = "gemma4:31b"
VISION_MODEL_DEFAULT = "gemma4:31b" 
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat_history.db"
AVATAR_PATH = BASE_DIR / "assistant.jpg"
LOGO_PATH = BASE_DIR / "logo.png"
OUTPUT_DIR = BASE_DIR / "generated_files"
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

def get_ollama_client():
    """Create an Ollama Cloud client using Streamlit Secrets."""
    api_key = st.secrets.get("OLLAMA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. Add it in Streamlit Cloud → Settings → Secrets."
        )
    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )


def ollama_generate(model, prompt, system_instruction=None, images=None):
    """Generate a response using Ollama Cloud."""
    client = get_ollama_client()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    user_message = {"role": "user", "content": str(prompt)}
    if images:
        user_message["images"] = images
    messages.append(user_message)
    response = client.chat(
        model=(model or TEXT_MODEL_DEFAULT).strip(),
        messages=messages,
        stream=False,
    )
    message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return (content or "").strip()


st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- DATABASE -----------------------------
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)""")
        conn.commit()


def new_conversation():
    cid = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as conn:
        conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (cid, "New conversation", now, now))
        conn.commit()
    return cid


def get_conversations(search=""):
    with db_connect() as conn:
        if search.strip():
            term = f"%{search.strip()}%"
            rows = conn.execute("""SELECT DISTINCT c.id, c.title, c.updated_at
                FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id
                WHERE c.title LIKE ? OR m.content LIKE ? ORDER BY c.updated_at DESC""", (term, term)).fetchall()
        else:
            rows = conn.execute("SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_messages(cid):
    with db_connect() as conn:
        rows = conn.execute("SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id", (cid,)).fetchall()
    return [dict(r) for r in rows]


def save_message(cid, role, content):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as conn:
        conn.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)", (cid, role, content, now))
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, cid))
        conn.commit()


def set_title(cid, title):
    title = " ".join(title.split()).strip()[:55] or "New conversation"
    with db_connect() as conn:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
        conn.commit()


def delete_conversation(cid):
    with db_connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
        conn.commit()


def clear_history():
    with db_connect() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        conn.commit()


init_db()

# ----------------------------- HELPERS -----------------------------
def avatar_path():
    return str(AVATAR_PATH) if AVATAR_PATH.exists() else "🤖"


def get_installed_models():
    return ["gemma4:31b"]

def read_uploaded_file(uploaded_file):
    """Extract readable text from supported uploaded documents."""
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix in {".txt", ".md", ".py", ".java", ".js", ".ts", ".html", ".css", ".json", ".xml", ".yml", ".yaml", ".sql", ".log", ".sh", ".bat"}:
        return raw.decode("utf-8", errors="replace")
    if suffix == ".csv":
        return raw.decode("utf-8-sig", errors="replace")
    if suffix == ".pdf":
        try:
            import importlib
            PdfReader = importlib.import_module("pypdf").PdfReader
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for idx, page in enumerate(reader.pages[:40], start=1):
                pages.append(f"[Page {idx}]\n{page.extract_text() or '[No selectable text on this page]'}")
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError("PDF reading requires pypdf. Install it with: py -3.12 -m pip install pypdf")
    raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")


def trim_text(text, max_chars=24000):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Content shortened to fit the local model context.]"


def safe_filename(name):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    cleaned = "".join(ch if ch in allowed else "_" for ch in name).strip("_")
    return cleaned[:70] or "generated_file"


def save_generated_text(filename, content):
    suffix = Path(filename).suffix or ".txt"
    stem = safe_filename(Path(filename).stem)
    path = OUTPUT_DIR / f"{stem}{suffix}"
    counter = 1
    while path.exists():
        path = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    path.write_text(content, encoding="utf-8")
    return path


# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp{background:radial-gradient(ellipse at 8% 0%,rgba(36,83,175,.22),transparent 35%),radial-gradient(ellipse at 95% 10%,rgba(120,50,210,.16),transparent 32%),#080b16;color:#eef3ff;font-family:Inter,sans-serif}
[data-testid="stHeader"]{background:rgba(8,11,22,.7)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101629,#090f1e);border-right:1px solid #263453}
[data-testid="stSidebar"] img{object-fit:contain}
.hero{padding:26px 28px;border:1px solid rgba(85,231,255,.27);border-radius:22px;background:linear-gradient(125deg,rgba(21,34,65,.96),rgba(32,20,60,.88));box-shadow:0 0 30px rgba(85,231,255,.06);margin-bottom:22px}
.eyebrow{color:#55e7ff;font-size:11px;letter-spacing:3px;font-weight:800}
.hero h1{margin:8px 0;font-size:clamp(28px,4vw,42px);font-weight:800;letter-spacing:-1px;background:linear-gradient(90deg,#fff,#8ceeff,#c4a7ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:#aab8d8;margin:0;font-size:14px}
.pill{display:inline-block;padding:5px 10px;border:1px solid #28556b;border-radius:20px;color:#80edff;font-size:10px;margin-top:14px}
.section{color:#8b9aba;font-size:11px;letter-spacing:2px;font-weight:800;margin:18px 0 8px}
div[data-testid="stChatMessage"]{background:rgba(18,26,47,.8);border:1px solid rgba(146,166,213,.14);border-radius:17px;padding:13px 16px;margin-bottom:10px}
/* Larger profile avatars */
div[data-testid="stChatMessage"] [data-testid*="Avatar"]{width:3.4rem!important;height:3.4rem!important;min-width:3.4rem!important;min-height:3.4rem!important}
div[data-testid="stChatMessage"] img{width:3.4rem!important;height:3.4rem!important;min-width:3.4rem!important;object-fit:cover!important;border-radius:50%!important}
div[data-testid="stChatInput"]{border:1px solid rgba(85,231,255,.4);border-radius:16px;background:#11182b}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid rgba(85,231,255,.25);background:linear-gradient(135deg,#172640,#201b3d);color:#eafaff;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#55e7ff;color:#55e7ff}
.feature{padding:15px;min-height:100px;border-radius:15px;background:rgba(17,26,48,.85);border:1px solid rgba(143,160,208,.16);margin-bottom:8px}
.feature b{color:#dbe8ff}.feature span{display:block;color:#8fa0c4;font-size:12px;margin-top:7px}
hr{border-color:rgba(145,160,207,.13)} footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ----------------------------- SESSION STATE -----------------------------
for key, default in {
    "cid": None, "delete_id": None, "clear_confirm": False,
    "pending_prompt": None, "last_generated_image": None, "last_generated_file": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=86)
    st.markdown("""<div style="padding:4px 4px 16px"><div style="font-size:11px;letter-spacing:3px;color:#55e7ff;font-weight:800">M / AI LABS</div><div style="font-size:23px;font-weight:800;margin-top:5px">Marvellous <span style="color:#55e7ff">AI</span></div><div style="font-size:11px;color:#8493b6;margin-top:5px">AI WORKSPACE</div></div>""", unsafe_allow_html=True)

    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.cid = new_conversation()
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown('<div class="section">SEARCH HISTORY</div>', unsafe_allow_html=True)
    search = st.text_input("Search history", placeholder="Search questions / answers...", label_visibility="collapsed")
    conversations = get_conversations(search)
    st.markdown('<div class="section">YOUR CONVERSATIONS</div>', unsafe_allow_html=True)
    if not conversations:
        st.caption("No conversations found.")
    for item in conversations:
        col_chat, col_delete = st.columns([5, 1])
        with col_chat:
            prefix = "▸ " if item["id"] == st.session_state.cid else "◦ "
            if st.button(prefix + item["title"][:25], key=f"open_{item['id']}", use_container_width=True):
                st.session_state.cid = item["id"]
                st.session_state.delete_id = None
                st.rerun()
        with col_delete:
            if st.button("🗑", key=f"trash_{item['id']}", help="Delete this chat"):
                st.session_state.delete_id = item["id"]
        if st.session_state.delete_id == item["id"]:
            st.warning(f'Delete "{item["title"]}" permanently?')
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("Delete", key=f"yes_{item['id']}"):
                    delete_conversation(item["id"])
                    if st.session_state.cid == item["id"]:
                        st.session_state.cid = None
                    st.session_state.delete_id = None
                    st.rerun()
            with no_col:
                if st.button("Cancel", key=f"no_{item['id']}"):
                    st.session_state.delete_id = None
                    st.rerun()

    st.divider()
    text_model = st.text_input("Text model", value=TEXT_MODEL_DEFAULT, help="Ollama Cloud model name. Check ollama.com/search?c=cloud for available models.")
    vision_model = st.text_input("Vision model", value=VISION_MODEL_DEFAULT, help="Choose a vision-capable Ollama Cloud model for image understanding.")
    if st.button("Clear ALL history", use_container_width=True):
        st.session_state.clear_confirm = True
    if st.session_state.clear_confirm:
        st.warning("All saved chats will be permanently deleted.")
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("Confirm clear", key="clear_yes"):
                clear_history()
                st.session_state.cid = None
                st.session_state.clear_confirm = False
                st.session_state.delete_id = None
                st.rerun()
        with no_col:
            if st.button("Cancel", key="clear_no"):
                st.session_state.clear_confirm = False
                st.rerun()
    st.caption("Chat history uses the app runtime filesystem and may reset after cloud restarts.")

# ----------------------------- MAIN HEADER -----------------------------
st.markdown("""<div class="hero"><div class="eyebrow">✦ YOUR PERSONAL AI WORKSPACE</div><h1>आम्ही Technical संस्कार करतो  !!!</h1><p>Marvellous AI — your Python tutor, coding partner, file reader and creative assistant.</p><span class="pill">● OLLAMA CLOUD</span> &nbsp; <span class="pill">CHAT MEMORY</span> &nbsp; <span class="pill">IMAGE + FILE TOOLS</span></div>""", unsafe_allow_html=True)

chat_tab, upload_tab, image_tab, file_tab, about_tab = st.tabs(["💬 Chat", "📎 Upload & Analyze", "🎨 Generate Image", "🧾 Generate File", "ℹ️ About Us"])

# ----------------------------- CHAT TAB -----------------------------
with chat_tab:
    if st.session_state.cid is None:
        st.markdown("### What are we building today?")
        st.caption("Choose a prompt or type your own question below.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="feature"><b>⌘ Python Tutor</b><span>Learn Python with simple examples.</span></div>', unsafe_allow_html=True)
            if st.button("Learn Python", use_container_width=True):
                st.session_state.cid = new_conversation()
                st.session_state.pending_prompt = "Teach me Python programming from the basics with an example."
                st.rerun()
        with c2:
            st.markdown('<div class="feature"><b>⚡ Debug Code</b><span>Understand errors and fix code.</span></div>', unsafe_allow_html=True)
            if st.button("Debug my code", use_container_width=True):
                st.session_state.cid = new_conversation()
                st.session_state.pending_prompt = "Explain how to debug a Python program step by step."
                st.rerun()
        with c3:
            st.markdown('<div class="feature"><b>◈ Explore AI</b><span>Learn AI and deep learning basics.</span></div>', unsafe_allow_html=True)
            if st.button("Explore AI", use_container_width=True):
                st.session_state.cid = new_conversation()
                st.session_state.pending_prompt = "Explain artificial intelligence in beginner-friendly language."
                st.rerun()
    else:
        for msg in get_messages(st.session_state.cid):
            with st.chat_message(msg["role"], avatar=avatar_path() if msg["role"] == "assistant" else "🧑‍💻"):
                st.markdown(msg["content"])
                st.caption(msg["created_at"])

    typed_prompt = st.chat_input("Ask Marvellous AI anything... (Enter to send)", key="chat_prompt")
    prompt = typed_prompt or st.session_state.pending_prompt
    if st.session_state.pending_prompt is not None and prompt == st.session_state.pending_prompt:
        st.session_state.pending_prompt = None

    if prompt and prompt.strip():
        prompt = prompt.strip()
        if st.session_state.cid is None:
            st.session_state.cid = new_conversation()
        cid = st.session_state.cid
        if not get_messages(cid):
            set_title(cid, prompt)
        save_message(cid, "user", prompt)
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=avatar_path()):
            status = st.empty()
            status.info(f"Connecting to Ollama Cloud model `{text_model}`…")
        try:
            history = get_messages(cid)
            system_instruction = (
                """
                You are Marvellous AI, an expert AI assistant and programming teacher.

                    RESPONSE RULES:
                    1. Give detailed, complete and useful answers. Do not give overly short answers.
                    2. For normal questions, explain the answer with examples when useful.
                    3. For technical questions, explain what it is, how it works, and why it is used.
                    4. For programming questions, provide complete working code when requested, then explain the important parts.
                    5. For comparison questions, explain the differences in a table when appropriate.
                    6. For step-by-step tasks, provide numbered steps in the correct order.
                    7. For simple questions, answer concisely; for complex questions, provide a detailed explanation.
                    8. For Hindi or Hinglish questions, reply in simple Hindi or Hinglish.
                    9. Use headings, bullet points, and Markdown to make long answers easy to read.
                    10. Do not omit important details merely to keep the answer short.
                    11. If the user asks for complete code, provide the complete code rather than only a snippet.
                    12. Never claim to have tested code unless you actually tested it.
                    13. if the user asks for a tutorial, provide a clear step-by-step tutorial with examples.

                    Your goal is to teach the user clearly, not just provide a one-line answer.
            """
            )
            messages = [{"role": "system", "content": system_instruction}]
            for m in history:
                messages.append({"role": m["role"], "content": m["content"]})

            status.info("Ollama Cloud is generating your answer…")
            answer_box = st.empty()
            answer_parts = []
            client = get_ollama_client()
            stream = client.chat(
                    model=text_model.strip() or TEXT_MODEL_DEFAULT,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": 0.7,
                        "num_predict": 4096
                    }
            )

            for chunk in stream:
                if isinstance(chunk, dict):
                    piece = chunk.get("message", {}).get("content", "")
                else:
                    piece = getattr(getattr(chunk, "message", None), "content", "")
                if piece:
                    answer_parts.append(piece)
                    answer_box.markdown("".join(answer_parts) + " ▌")

            answer = "".join(answer_parts).strip()
            if not answer:
                raise RuntimeError("Ollama Cloud returned an empty response.")

            save_message(cid, "assistant", answer)
            status.empty()
            answer_box.markdown(answer)

            with st.expander("Copy answer"):
                st.code(answer, language=None)
        except Exception as exc:
            answer = (f"I could not get an answer from Ollama Cloud. Check OLLAMA_API_KEY in Streamlit Secrets and confirm the cloud model `{text_model}` is available.\n\nTechnical error: `{type(exc).__name__}: {exc}`")
            save_message(cid, "assistant", answer)
            status.empty()
            st.error(answer)
        st.rerun()

# ----------------------------- UPLOAD & ANALYZE TAB -----------------------------
with upload_tab:
    st.markdown("### Upload an image or document")
    st.caption("Documents are analyzed using Ollama Cloud. Image analysis requires a vision-capable Ollama Cloud model.")
    uploads = st.file_uploader(
        "Choose images or documents", type=["png", "jpg", "jpeg", "webp", "pdf", "txt", "md", "csv", "py", "java", "js", "ts", "html", "css", "json", "xml", "yml", "yaml", "sql", "log", "sh", "bat"],
        accept_multiple_files=True, key="analysis_uploads"
    )
    if uploads:
        for uploaded in uploads:
            suffix = Path(uploaded.name).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                st.image(uploaded, caption=uploaded.name, width=240)
            else:
                st.write(f"📄 {uploaded.name} · {len(uploaded.getvalue()):,} bytes")
        analysis_prompt = st.text_area("What should Marvellous AI do with these files?", value="Summarize the uploaded content and explain the important points.", key="analysis_prompt")
        if st.button("Analyze uploaded files", type="primary", key="analyze_button"):
            image_files = [f for f in uploads if Path(f.name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
            doc_files = [f for f in uploads if f not in image_files]
            if image_files and len(image_files) > 1:
                st.info("Image analysis will use the first uploaded image in this run. You can analyze more images one at a time.")
                image_files = image_files[:1]
            context_parts = []
            try:
                with st.spinner("Reading files and asking Ollama Cloud…"):
                    for file in doc_files:
                        context_parts.append(f"\n\n--- FILE: {file.name} ---\n{trim_text(read_uploaded_file(file))}")
                    full_prompt = analysis_prompt.strip() or "Analyze these files."
                    if context_parts:
                        full_prompt += "\n\nDocument contents:\n" + "\n".join(context_parts)
                    if image_files:
                        vision_name = vision_model.strip()
                        image_file = image_files[0]
                        image_b64 = base64.b64encode(image_file.getvalue()).decode("utf-8")
                        result_text = ollama_generate(
                            vision_name,
                            full_prompt + ("\n\nAlso consider these document contents." if context_parts else ""),
                            "Analyze the provided image carefully. Reply in the user's language and keep the answer useful and concise.",
                            images=[image_b64],
                        )
                    else:
                        result_text = ollama_generate(
                            text_model.strip(),
                            full_prompt,
                            "You carefully analyze user-provided files. Be accurate, concise, and reply in the user's language."
                        )
                st.markdown("#### Analysis result")
                st.markdown(result_text or "The model returned an empty response.")
                st.download_button("Download analysis as TXT", data=(result_text or ""), file_name="marvellous_analysis.txt", mime="text/plain")
            except Exception as exc:
                st.error(f"Could not analyze the upload: {type(exc).__name__}: {exc}")
                if image_files:
                    st.info(f"Image analysis requires a vision-capable Ollama Cloud model. Check OLLAMA_API_KEY and the selected model capabilities.")
                if any(Path(f.name).suffix.lower() == ".pdf" for f in doc_files):
                    st.info("For PDFs, install pypdf: `py -3.12 -m pip install pypdf`")

# ----------------------------- LOCAL IMAGE GENERATION TAB -----------------------------
with image_tab:
    st.markdown("### Local image generation studio")
    st.caption("Images are generated on your computer with Diffusers. The first run downloads model files. CPU generation can be very slow; a supported NVIDIA GPU is faster.")
    with st.expander("One-time setup", expanded=False):
        st.code("py -3.12 -m pip install torch diffusers transformers accelerate safetensors", language="powershell")
        st.write("This app uses the small `segmind/tiny-sd` model by default to make local setup lighter. It still needs an internet connection for the first model download, then runs locally from cache. This feature is separate from Ollama Cloud and may not run reliably on Streamlit Cloud.")
    image_prompt = st.text_area("Describe the image you want", placeholder="A futuristic blue and purple cyberpunk city at night, cinematic lighting, detailed digital art", height=100, key="image_prompt")
    col1, col2 = st.columns(2)
    with col1:
        image_steps = st.slider("Inference steps", min_value=1, max_value=8, value=4, help="More steps can improve results but take longer.")
    with col2:
        image_seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2147483647, value=0, step=1)
    if st.button("✨ Generate image locally", type="primary", key="generate_image_button"):
        if not image_prompt.strip():
            st.warning("Enter an image description first.")
        else:
            try:
                import torch
                import importlib
                AutoPipelineForText2Image = importlib.import_module("diffusers").AutoPipelineForText2Image
                model_id = "segmind/tiny-sd"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                with st.spinner(f"Loading local image model on {device.upper()}… first run may download several GB and take time."):
                    @st.cache_resource(show_spinner=False)
                    def load_image_pipeline(model_name, device_name, dtype_name):
                        import torch as _torch
                        import importlib as _importlib
                        _Pipeline = _importlib.import_module("diffusers").AutoPipelineForText2Image
                        dtype_local = _torch.float16 if dtype_name == "float16" else _torch.float32
                        pipe = _Pipeline.from_pretrained(model_name, torch_dtype=dtype_local)
                        pipe = pipe.to(device_name)
                        if device_name == "cpu":
                            pipe.enable_attention_slicing()
                        return pipe
                    pipe = load_image_pipeline(model_id, device, "float16" if device == "cuda" else "float32")
                generator = torch.Generator(device=device).manual_seed(int(image_seed)) if image_seed else None
                with st.spinner("Creating your image… CPU may take several minutes."):
                    result = pipe(prompt=image_prompt.strip(), num_inference_steps=int(image_steps), guidance_scale=0.0, generator=generator)
                    generated = result.images[0]
                out_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}.png"
                out_path = IMAGE_DIR / out_name
                generated.save(out_path)
                st.session_state.last_generated_image = str(out_path)
                st.success("Image generated and saved locally.")
                st.image(generated, caption=image_prompt.strip(), use_container_width=True)
                st.download_button("Download generated PNG", data=out_path.read_bytes(), file_name=out_name, mime="image/png")
            except ImportError as exc:
                st.error(f"Image generation packages are missing: {exc}")
                st.code("py -3.12 -m pip install torch diffusers transformers accelerate safetensors", language="powershell")
            except Exception as exc:
                st.error(f"Local image generation failed: {type(exc).__name__}: {exc}")
                st.info("Try again with enough free disk space and memory. If your computer has no supported GPU, CPU generation may be very slow. Model downloads require internet on the first run.")
    if st.session_state.last_generated_image and Path(st.session_state.last_generated_image).exists():
        st.markdown("#### Last generated image")
        st.image(st.session_state.last_generated_image, use_container_width=True)
        st.download_button("Download last image", data=Path(st.session_state.last_generated_image).read_bytes(), file_name=Path(st.session_state.last_generated_image).name, mime="image/png", key="download_last_image")

# ----------------------------- FILE GENERATION TAB -----------------------------
with file_tab:
    st.markdown("### Generate a file from your prompt")
    st.caption("Ask Ollama Cloud to create content, then save it as a downloadable file. Review generated code before running it.")
    file_prompt = st.text_area("What file should the AI create?", placeholder="Create a beginner-friendly Python calculator with comments", height=110, key="file_prompt")
    file_type = st.selectbox("Output format", ["Python (.py)", "Text (.txt)", "Markdown (.md)", "CSV (.csv)", "JSON (.json)", "HTML (.html)", "SQL (.sql)"])
    default_names = {"Python (.py)": "generated_code.py", "Text (.txt)": "generated_text.txt", "Markdown (.md)": "generated_notes.md", "CSV (.csv)": "generated_data.csv", "JSON (.json)": "generated_data.json", "HTML (.html)": "generated_page.html", "SQL (.sql)": "generated_query.sql"}
    file_name = st.text_input("File name", value=default_names[file_type])
    if st.button("🧾 Generate file", type="primary", key="generate_file_button"):
        if not file_prompt.strip():
            st.warning("Describe the file you want first.")
        else:
            ext = Path(file_name).suffix.lower()
            allowed_exts = {"Python (.py)": ".py", "Text (.txt)": ".txt", "Markdown (.md)": ".md", "CSV (.csv)": ".csv", "JSON (.json)": ".json", "HTML (.html)": ".html", "SQL (.sql)": ".sql"}
            expected_ext = allowed_exts[file_type]
            if ext != expected_ext:
                file_name = str(Path(file_name).with_suffix(expected_ext))
            format_instruction = {
                ".py": "Return only valid Python source code, no Markdown fences.",
                ".txt": "Return plain text only.",
                ".md": "Return well-structured Markdown.",
                ".csv": "Return only valid CSV with a header row if appropriate. Do not use Markdown fences.",
                ".json": "Return only valid JSON. Do not use Markdown fences.",
                ".html": "Return a complete HTML document. Do not use Markdown fences.",
                ".sql": "Return SQL only, with brief SQL comments if useful. Do not use Markdown fences.",
            }[expected_ext]
            try:
                with st.spinner(f"Asking Ollama Cloud ({text_model.strip()}) to create your file…"):
                    content = ollama_generate(
                        text_model.strip(),
                        f"Create the requested file. {format_instruction}\n\nRequest: {file_prompt.strip()}",
                        "Generate useful, accurate file contents. Follow the requested format exactly."
                    )
                # Remove a surrounding Markdown fence if a model adds one anyway.
                if content.startswith("```"):
                    lines = content.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = "\n".join(lines).strip()
                out_path = save_generated_text(file_name, content)
                st.session_state.last_generated_file = str(out_path)
                st.success(f"File created: {out_path.name}")
                st.code(content[:15000], language={".py": "python", ".json": "json", ".html": "html", ".sql": "sql", ".csv": "csv", ".md": "markdown"}.get(expected_ext, "text"))
                st.download_button("⬇ Download generated file", data=out_path.read_bytes(), file_name=out_path.name, mime="text/plain")
            except Exception as exc:
                st.error(f"File generation failed: {type(exc).__name__}: {exc}")
                st.info(f"Check OLLAMA_API_KEY in Streamlit Secrets, your Ollama Cloud usage access, internet access, and the selected cloud model name.")
    if st.session_state.last_generated_file and Path(st.session_state.last_generated_file).exists():
        last_file = Path(st.session_state.last_generated_file)
        st.caption(f"Last generated file: {last_file.name}")
        st.download_button("Download last generated file", data=last_file.read_bytes(), file_name=last_file.name, mime="application/octet-stream", key="download_last_file")

st.markdown("<div style='text-align:center;color:#667596;font-size:11px;padding:22px 0 8px'>MARVELLOUS AI · STREAMLIT + OLLAMA CLOUD + LOCAL DIFFUSERS<br><br>AI can make mistakes. Review generated content and code before using it.</div>", unsafe_allow_html=True)

# ----------------------------- ABOUT US TAB -----------------------------
with about_tab:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">✦ ABOUT THIS PROJECT</div>
      <h1>About <span style="color:#55e7ff">Marvellous AI</span></h1>
      <p>A personal AI workspace for learning, coding, and creating.</p>
      <span class="pill">OLLAMA CLOUD</span> &nbsp; <span class="pill">PYTHON + STREAMLIT</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 👨‍💻 Created by Sachin Patel")
    st.write(
        "Marvellous AI is a personal AI assistant project created by Sachin Patel. "
        "It combines AI chat, programming assistance, supported document and image analysis, "
        "and file-generation tools in one Streamlit application."
    )
    st.divider()
    st.markdown("### 🏢 About Marvellous Infosystems")
    st.write(
        "Marvellous Infosystems is a Pune-based IT and computer training institute. "
        "Visit its official website for current information about courses, training programs, and learning resources."
    )
    st.markdown("🌐 Official website: [www.marvellousinfosystems.com](https://www.marvellousinfosystems.com/)")
    st.divider()
    st.markdown("### ✨ App features")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**💬 AI Chat & Coding**")
        st.caption("Ask questions and get programming help.")
        st.markdown("**📎 Upload & Analyze**")
        st.caption("Analyze supported documents and images with Ollama Cloud.")
    with c2:
        st.markdown("**🎨 Generate Image**")
        st.caption("Generate images locally when required packages and resources are available.")
        st.markdown("**🧾 Generate File**")
        st.caption("Generate code and supported text-based files from prompts.")
    st.caption("Independent project created by Sachin Patel; not an official Marvellous Infosystems product.")
