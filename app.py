import streamlit as st
import os
from rag_engine import RAGEngine
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="ML Learning Assistant",
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
    .sidebar .sidebar-content {
        background-color: rgba(15, 23, 42, 0.8);
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

# Load environment variables
load_dotenv()

def initialize_engine():
    if "engine" not in st.session_state:
        pdf_file = "Machine Learning By Thiru Book.pdf"
        if not os.path.exists(pdf_file):
            st.error(f"Error: {pdf_file} not found.")
            return None
        
        try:
            engine = RAGEngine(pdf_file)
            engine.initialize()
            st.session_state.engine = engine
        except Exception as e:
            st.error(f"Failed to initialize RAG Engine: {e}")
            return None
    return st.session_state.engine

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
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
    st.info("""
    **ML Learning Assistant**
    
    This tool uses RAG to help you learn Machine Learning from 'Machine Learning By Thiru Book'.
    
    Ask questions like:
    - What is Supervised Learning?
    - Explain Linear Regression.
    - How does k-NN work?
    """)

# Main Content
st.title("🤖 Machine Learning Assistant")
st.subheader("Your AI-powered study buddy for Thiru's ML Book")

if not os.getenv("GOOGLE_API_KEY"):
    st.info("Please provide your Gemini API Key in the sidebar or a .env file to get started.")
else:
    engine = initialize_engine()
    
    if engine:
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about Machine Learning..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching book and thinking..."):
                    try:
                        response = engine.query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Powered by LangChain, ChromaDB, and Gemini 2.0 Flash")
