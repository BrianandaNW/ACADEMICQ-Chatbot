import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AkademiQ — Asisten Cerdas",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# INJEKSI CSS — TEMA "AURORA ACADEMIA"
# ==========================================
st.markdown("""
<style>
/* ---- IMPORT FONT ---- */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --aurora-1: #FF6B9D;
    --aurora-2: #C44DFF;
    --aurora-3: #4DC8FF;
    --aurora-4: #FFB347;
    --aurora-5: #43E97B;
    --card-bg: rgba(255, 255, 255, 0.72);
    --text-dark: #1a1a2e;
    --text-mid: #3d3d6b;
    --shadow-soft: 0 8px 32px rgba(196, 77, 255, 0.15);
    --shadow-card: 0 4px 24px rgba(0,0,0,0.10);
}

/* ---- BACKGROUND AURORA ---- */
.stApp {
    background: linear-gradient(135deg,
        #ffecd2 0%,
        #fcb69f 15%,
        #ffeaa7 30%,
        #dfe9f3 45%,
        #c3cfe2 60%,
        #e0c3fc 75%,
        #f093fb 90%,
        #f5576c 100%
    ) !important;
    background-attachment: fixed !important;
    font-family: 'Inter', sans-serif;
}

/* ---- ANIMATED FLOATING BLOBS ---- */
.stApp::before {
    content: "";
    position: fixed;
    top: -20%;
    left: -10%;
    width: 60vw;
    height: 60vw;
    background: radial-gradient(circle, rgba(255,107,157,0.28) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatBlob1 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: "";
    position: fixed;
    bottom: -20%;
    right: -10%;
    width: 55vw;
    height: 55vw;
    background: radial-gradient(circle, rgba(77,200,255,0.22) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatBlob2 15s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes floatBlob1 {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(5%, 8%) scale(1.1); }
}
@keyframes floatBlob2 {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(-6%, -5%) scale(1.08); }
}

/* ---- SIDEBAR MEWAH ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(255,255,255,0.92) 0%,
        rgba(240,230,255,0.95) 50%,
        rgba(255,255,255,0.92) 100%) !important;
    backdrop-filter: blur(20px) saturate(180%);
    border-right: 1.5px solid rgba(196,77,255,0.18) !important;
    box-shadow: 4px 0 30px rgba(196,77,255,0.10);
}
[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

/* ---- HEADER TRANSPARAN ---- */
[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}

/* ---- MAIN CONTENT AREA ---- */
[data-testid="stMain"] .block-container {
    padding-top: 1.5rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ---- JUDUL UTAMA ---- */
.akademiq-hero {
    background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #f5576c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.15;
    margin-bottom: 0;
}
.akademiq-tagline {
    font-family: 'Inter', sans-serif;
    color: #5b4a7e;
    font-size: 1.05rem;
    font-weight: 500;
    margin-top: 0.35rem;
}

/* ---- HERO CARD (Welcome Banner) ---- */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(102,126,234,0.35);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "✦ ✧ ✦";
    position: absolute;
    top: 12px;
    right: 20px;
    font-size: 1.4rem;
    opacity: 0.35;
    letter-spacing: 6px;
}
.hero-banner h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    margin: 0 0 0.4rem 0;
    color: white !important;
}
.hero-banner p {
    font-size: 0.9rem;
    opacity: 0.88;
    margin: 0;
    color: rgba(255,255,255,0.92) !important;
}

/* ---- FEATURE CHIPS ---- */
.chips-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 1rem 0 1.4rem 0;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(12px);
    border: 1.5px solid rgba(196,77,255,0.22);
    border-radius: 50px;
    padding: 6px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #5b3f9a;
    box-shadow: 0 2px 10px rgba(196,77,255,0.12);
    transition: all 0.2s ease;
}
.chip:hover {
    background: rgba(196,77,255,0.12);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(196,77,255,0.22);
}

/* ---- CHAT MESSAGES ---- */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(16px) saturate(160%) !important;
    border: 1px solid rgba(255,255,255,0.65) !important;
    border-radius: 18px !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.10) !important;
    padding: 0.8rem 1.1rem !important;
    transition: box-shadow 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 6px 28px rgba(102,126,234,0.18) !important;
}

/* User message — warna aksen ungu muda */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) {
    border-left: 3px solid transparent;
}

/* ---- CHAT INPUT ---- */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.88) !important;
    border-radius: 28px !important;
    border: 2px solid rgba(196,77,255,0.25) !important;
    box-shadow: 0 8px 30px rgba(196,77,255,0.15), 0 2px 8px rgba(0,0,0,0.06) !important;
    backdrop-filter: blur(16px) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(196,77,255,0.55) !important;
    box-shadow: 0 8px 32px rgba(196,77,255,0.30) !important;
}

/* ---- SIDEBAR CONTENT ---- */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.8rem;
}
.sidebar-logo-text {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    background: linear-gradient(90deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.sidebar-logo-sub {
    font-size: 0.72rem;
    color: #8a7aaa;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
}

.stat-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(240,147,251,0.12));
    border: 1px solid rgba(196,77,255,0.18);
    border-radius: 14px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.stat-icon { font-size: 1.4rem; }
.stat-label {
    font-size: 0.78rem;
    color: #8a7aaa;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    margin: 0;
}
.stat-value {
    font-size: 1rem;
    font-weight: 700;
    color: #4a3a7a;
    font-family: 'Poppins', sans-serif;
    margin: 0;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 10px 0;
}
.step-num {
    background: linear-gradient(135deg, #667eea, #f093fb);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    font-size: 0.85rem;
    color: #4a3a6a;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    padding-top: 3px;
}

.divider-gradient {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(196,77,255,0.35), transparent);
    border: none;
    margin: 14px 0;
}

/* ---- STATUS BADGE ---- */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: linear-gradient(90deg, rgba(67,233,123,0.15), rgba(67,200,255,0.15));
    border: 1px solid rgba(67,233,123,0.4);
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #1a6e45;
    font-family: 'Inter', sans-serif;
}
.dot-pulse {
    width: 8px;
    height: 8px;
    background: #43e97b;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
}

/* ---- FIX: SEMBUNYIKAN TEKS IKON ARROW PADA EXPANDER ---- */
[data-testid="stExpander"] details summary span span {
    font-size: 0 !important;
    color: transparent !important;
    visibility: hidden !important;
    width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details > summary > span > span {
    font-size: 0 !important;
    color: transparent !important;
    visibility: hidden !important;
}

/* ---- EXPANDER ---- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.65) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(196,77,255,0.18) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

/* ---- SPINNER ---- */
[data-testid="stSpinner"] {
    color: #764ba2 !important;
}

/* ---- FIX: SEMBUNYIKAN TEKS "keyboard_double" PADA TOMBOL SIDEBAR ---- */
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
section[data-testid="stSidebar"] + div button span {
    font-size: 0 !important;
    visibility: hidden !important;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #c44dff, #667eea);
    border-radius: 6px;
}

/* ---- GENERAL STREAMLIT OVERRIDES ---- */
h1, h2, h3 {
    font-family: 'Poppins', sans-serif !important;
    color: #2c1f5e !important;
}
p, li, span {
    font-family: 'Inter', sans-serif;
    color: #3d3460;
}
.stMarkdown p {
    color: #3d3460;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# KONFIGURASI API
# ==========================================
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ==========================================
# LOGIKA RAG & BASIS DATA
# ==========================================
@st.cache_resource
def load_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def setup_rag_chain():
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    system_prompt = (
        "Kamu adalah asisten akademik yang cerdas, ramah, dan sangat membantu. "
        "Gunakan HANYA potongan konteks dokumen berikut untuk menjawab pertanyaan. "
        "Jika jawabannya tidak ada di dalam konteks, katakan saja dengan sopan bahwa "
        "informasi tersebut tidak ada di dalam dokumen panduan. "
        "Jangan mengarang jawaban."
        "\n\n"
        "Konteks Dokumen:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# ==========================================
# SIDEBAR — "AURORA ACADEMIA"
# ==========================================
with st.sidebar:

    # Logo & Brand
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 6px 0;">
        <div style="font-size: 3.2rem; margin-bottom: 4px;">🎓</div>
        <div class="sidebar-logo-text">AkademiQ</div>
        <div class="sidebar-logo-sub">Asisten Panduan Akademik</div>
    </div>
    """, unsafe_allow_html=True)

    # Status Online
    st.markdown("""
    <div style="text-align:center; margin: 10px 0 14px 0;">
        <span class="status-online">
            <span class="dot-pulse"></span> Sistem Aktif & Siap
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-gradient">', unsafe_allow_html=True)

    # Statistik / Info Cards
    st.markdown("**📊 Informasi Sistem**")
    st.markdown("""
    <div class="stat-card">
        <span class="stat-icon">📚</span>
        <div>
            <p class="stat-label">Basis Pengetahuan</p>
            <p class="stat-value">Panduan Akademik</p>
        </div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">🤖</span>
        <div>
            <p class="stat-label">Model AI</p>
            <p class="stat-value">Gemini 2.5 Flash</p>
        </div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">⚡</span>
        <div>
            <p class="stat-label">Teknologi RAG</p>
            <p class="stat-value">FAISS + LangChain</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-gradient">', unsafe_allow_html=True)

    # Panduan Penggunaan
    st.markdown("**🗺️ Cara Menggunakan**")
    st.markdown("""
    <div class="step-item">
        <div class="step-num">1</div>
        <div class="step-text">Ketik pertanyaanmu di kolom chat di bawah.</div>
    </div>
    <div class="step-item">
        <div class="step-num">2</div>
        <div class="step-text">AI akan membaca & mencari dalam dokumen panduan.</div>
    </div>
    <div class="step-item">
        <div class="step-num">3</div>
        <div class="step-text">Dapatkan jawaban akurat berbasis data resmi!</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-gradient">', unsafe_allow_html=True)

    # Tips Bertanya
    with st.expander("Contoh Pertanyaan Cerdas"):
        st.markdown("""
        <div style="font-size:0.85rem; color:#5b3f9a; line-height:2;">
            <b style="font-size:0.9rem;">💡 Coba tanyakan ini:</b><br><br>
            🔹 <i>Apa syarat kelulusan sarjana?</i><br>
            🔹 <i>Berapa SKS minimal per semester?</i><br>
            🔹 <i>Bagaimana prosedur pengajuan cuti?</i><br>
            🔹 <i>Apa itu mata kuliah pilihan?</i><br>
            🔹 <i>Kapan batas pengisian KRS?</i>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-gradient">', unsafe_allow_html=True)

    # Footer Sidebar
    st.markdown("""
    <div style="text-align:center; font-size:0.75rem; color:#a0899e; padding-top:4px;">
        Dibuat dengan ❤️ untuk kemudahan akademik<br>
        <span style="font-weight:600; color:#8a7aaa;">© 2025 AkademiQ</span>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# HALAMAN UTAMA
# ==========================================

# --- Hero Title ---
st.markdown("""
<div class="akademiq-hero">✨ AkademiQ</div>
<div class="akademiq-tagline">Asisten Panduan Akademik Berbasis AI — Jawaban Tepat, Langsung dari Sumbernya</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Welcome Banner ---
st.markdown("""
<div class="hero-banner">
    <h3>🌟 Selamat Datang di AkademiQ!</h3>
    <p>
        Saya adalah asisten akademik cerdasmu yang didukung teknologi RAG (Retrieval-Augmented Generation).
        Saya membaca langsung dari dokumen panduan resmi fakultasmu — tidak mengarang, tidak menebak-nebak.
        Tanyakan apa saja seputar aturan, syarat, dan prosedur akademik!
    </p>
</div>
""", unsafe_allow_html=True)

# --- Feature Chips ---
st.markdown("""
<div class="chips-row">
    <div class="chip">📖 Berbasis Dokumen Resmi</div>
    <div class="chip">⚡ Respons Cepat</div>
    <div class="chip">🎯 Jawaban Akurat</div>
    <div class="chip">🔒 Tidak Mengarang</div>
    <div class="chip">🌐 Bahasa Indonesia</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# CHAT INTERFACE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! 👋 Saya **AkademiQ**, asisten panduan akademikmu.\n\nSilakan tanyakan apa pun seputar aturan akademik — mulai dari syarat kelulusan, pengisian KRS, prosedur cuti, hingga informasi lainnya yang ada di dalam panduan resmi. Saya siap membantu! 🎓✨",
        }
    ]

# Tampilkan riwayat pesan
for msg in st.session_state.messages:
    avatar_icon = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# Input pengguna
user_query = st.chat_input("💬 Ketik pertanyaanmu di sini... (contoh: Apa syarat kelulusan?)")

if user_query:
    # Tampilkan pesan user
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Proses & tampilkan respons
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("⏳ Membaca dokumen panduan..."):
            try:
                rag_chain = setup_rag_chain()
                response = rag_chain.invoke({"input": user_query})
                jawaban = response["answer"]
                st.markdown(jawaban)
                st.session_state.messages.append({"role": "assistant", "content": jawaban})
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan teknis: {e}")