import langchain
print(f"LangChain version: {langchain.__version__}")
try:
    from langchain.chains import create_retrieval_chain
    print("Successfully imported from langchain.chains")
except ImportError as e:
    print(f"Failed to import from langchain.chains: {e}")

try:
    from langchain.chains.retrieval import create_retrieval_chain
    print("Successfully imported from langchain.chains.retrieval")
except ImportError as e:
    print(f"Failed to import from langchain.chains.retrieval: {e}")

import langchain.chains
print(f"Available items in langchain.chains: {dir(langchain.chains)}")
