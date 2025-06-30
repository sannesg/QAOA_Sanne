# Loading API key
from dotenv import load_dotenv

load_dotenv()

# For interactive Q&A
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# To load the files from the folder
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import NotebookLoader

# To use the OpenAI LLM
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# To create a vector index of the split elements
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Connecting LLM to vectorstore for Q&A
from langchain.chains import RetrievalQA

import os
import glob

# print(os.getenv("OPENAI_API_KEY"))

# Use the LLM from OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)


# "." since I am in the same folder
repo_path = "."

# Load all .py files from the folder and to the directory
loader_py = DirectoryLoader(repo_path, glob="**/*.py")
docs_py = loader_py.load()

# Load all .md files from the folder and to the directory
loader_md = DirectoryLoader(repo_path, glob="**/*.md")
docs_md = loader_md.load()

# Load .txt files
loader_txt = DirectoryLoader(repo_path, glob="**/*.txt")
docs_txt = loader_txt.load()

# Load Jupyter notebooks
notebook_paths = glob.glob(os.path.join(repo_path, "**/*.ipynb"), recursive=True)

docs_ipynb = []
for nb_path in notebook_paths:
    loader_nb = NotebookLoader(nb_path)
    docs_ipynb.extend(loader_nb.load())

# Combine the loaded documents
documents = docs_py + docs_md + docs_txt + docs_ipynb

# print(f"Loaded {len(documents)} documents")

# Optional: Split into chunks for better processing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_split = splitter.split_documents(documents)

# Turn the split documents into vectors
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs_split, embedding)

# print(f"Split into {len(docs_split)} chunks")


# Create a prompt template for the LLM, a code chain and an input

"""
prompt = PromptTemplate(
    input_variables=["problem", "initial_state", "mixer"],
    template=code_gen_template)
    
    

code_chain = LLMChain(llm=llm, prompt=prompt)
"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

with open("context_for_LLM.txt", "r") as f:
    context_prompt = f.read()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    chain_type_kwargs={"verbose": True, "system_message": context_prompt},
)

chat_history = []

while True:
    query = input("Ask a question (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = qa_chain({"question": query})
    print("\nAnswer:")
    print(result["answer"])
    print("-" * 40)
