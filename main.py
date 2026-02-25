import os
import sys
from rag_engine import RAGEngine
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file.")
        print("Please create a .env file with GOOGLE_API_KEY=your_api_key")
        return

    pdf_file = "Machine Learning By Thiru Book.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found.")
        return

    print("Initializing RAG Engine... This might take a moment if it's the first time.")
    try:
        engine = RAGEngine(pdf_file)
        engine.initialize()
    except Exception as e:
        print(f"Failed to initialize RAG Engine: {e}")
        return

    print("\n" + "="*50)
    print("Welcome to Machine Learning Learning Assistant!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            question = input("You: ")
            if question.lower() in ["exit", "quit"]:
                break
            
            if not question.strip():
                continue

            print("\nThinking...", end="\r")
            answer = engine.query(question)
            print("Assistant:", answer)
            print("\n" + "-"*30 + "\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError occurred: {e}\n")

    print("\nGoodbye!")

if __name__ == "__main__":
    main()
