import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, pdf_path: str, persist_directory: str = "./faiss_index"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        self.vector_store = None
        self.retrieval_chain = None

    def initialize(self):
        """Processes the PDF and initializes the vector store and retrieval chain."""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}...")
            self.vector_store = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"Processing PDF: {self.pdf_path}...")
            loader = PyPDFLoader(self.pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            print(f"Creating vector store at {self.persist_directory}...")
            self.vector_store = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            self.vector_store.save_local(self.persist_directory)
        
        # Setup retrieval chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        system_prompt = (
            "You are an assistant for question-answering tasks based on the provided document. "
            "Use the following pieces of retrieved context to answer the question. "
            "If the answer is not in the context, say you don't know. "
            "Keep the answer concise and helpful."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    def query(self, question: str):
        """Queries the RAG system."""
        if not self.retrieval_chain:
            raise ValueError("RAG Engine not initialized. Call initialize() first.")
        
        response = self.retrieval_chain.invoke({"input": question})
        return response["answer"]
