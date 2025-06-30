# Loading API key
from dotenv import load_dotenv

load_dotenv()

# To load the files from the folder
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# To use the OpenAI LLM
from langchain_openai import ChatOpenAI

# To create a vector index of the split elements
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Connecting LLM to vectorstore for Q&A
from langchain.chains import RetrievalQA

import os

print(os.getenv("OPENAI_API_KEY"))

# Use the LLM from OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Adjust this to the relative or absolute path to the folder you want to load
repo_path = "."  # or "." if you're in the same folder

# Load all .py files from the folder
loader = DirectoryLoader(repo_path, glob="**/*.py")

# Load documents from that directory
documents = loader.load()

print(f"Loaded {len(documents)} documents")

# Optional: Split into chunks for better processing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_split = splitter.split_documents(documents)

# Turn the split documents into vectors
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs_split, embedding)

print(f"Split into {len(docs_split)} chunks")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff"
)

query = "What does the main function in this repo do?"
response = qa_chain.run(query)

print("Answer:")
print(response)
