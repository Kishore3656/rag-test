import streamlit as st
import os
import glob
from rag_engine import RAGEngine
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        color: white;
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatInputContainer {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 20px;
    }
    h1, h2, h3 {
        color: #60a5fa !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()


def get_available_pdfs():
    return glob.glob("*.pdf")


def initialize_engine(pdf_path: str):
    """Initialize or re-initialize the RAG engine for a given PDF."""
    engine_key = f"engine_{pdf_path}"
    if engine_key not in st.session_state:
        try:
            persist_dir = f"./faiss_index_{os.path.splitext(os.path.basename(pdf_path))[0]}"
            engine = RAGEngine(pdf_path, persist_directory=persist_dir)
            engine.initialize()
            st.session_state[engine_key] = engine
        except Exception as e:
            st.error(f"Failed to initialize RAG Engine: {e}")
            return None
    return st.session_state[engine_key]


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ GOOGLE_API_KEY missing!")
        new_key = st.text_input("Enter Gemini API Key", type="password")
        if new_key:
            os.environ["GOOGLE_API_KEY"] = new_key
            st.success("API Key set for this session!")
            st.rerun()
    else:
        st.success("✅ API Key active")

    st.divider()

    # PDF selection
    pdfs = get_available_pdfs()
    if pdfs:
        selected_pdf = st.selectbox("📄 Select PDF", pdfs)
    else:
        st.error("No PDF files found in the project directory.")
        selected_pdf = None

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info("Ask questions about the selected PDF document. The assistant will find relevant passages and answer based on the content.")


# Main Content
st.title("🤖 RAG Assistant")
if selected_pdf:
    st.subheader(f"Chatting with: `{selected_pdf}`")

if not os.getenv("GOOGLE_API_KEY"):
    st.info("Please provide your Gemini API Key in the sidebar or a `.env` file to get started.")
elif not selected_pdf:
    st.warning("No PDF files found. Place a PDF in the project directory and restart.")
else:
    with st.spinner(f"Loading `{selected_pdf}`..."):
        engine = initialize_engine(selected_pdf)

    if engine:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching and thinking..."):
                    try:
                        response = engine.query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Powered by LangChain, FAISS, and Gemini 2.0 Flash")
