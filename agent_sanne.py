# Loading API key
from dotenv import load_dotenv

load_dotenv()

# For interactive Q&A
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# To load the files from the folder
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NotebookLoader

# To use the OpenAI LLM
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# To create a vector index of the split elements
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

import os
import glob

# print(os.getenv("OPENAI_API_KEY"))

# Use the LLM from OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)


# "." because the files are in the same folder
repo_path = "."

# Load all .py files
loader_py = DirectoryLoader(repo_path, glob="**/*.py")
docs_py = loader_py.load()

# Load all .md files
loader_md = DirectoryLoader(repo_path, glob="**/*.md")
docs_md = loader_md.load()

# Load .txt files
loader_txt = DirectoryLoader(repo_path, glob="**/*.txt")
docs_txt = loader_txt.load()

# Load .iptnb files
notebook_paths = glob.glob(os.path.join(repo_path, "**/*.ipynb"), recursive=True)

docs_ipynb = []
for nb_path in notebook_paths:
    loader_nb = NotebookLoader(nb_path)
    docs_ipynb.extend(loader_nb.load())

# Combine the loaded documents
documents = docs_py + docs_md + docs_txt + docs_ipynb

# Loading the context_for_LLM.txt file as a string
with open("context_for_LLM.txt", "r", encoding="utf-8") as f:
    context_instructions = f.read()

# Build a prompt template object with the context instructions
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(context_instructions),
        HumanMessagePromptTemplate.from_template(
            "Here is some additional context:\n{context}\n\nNow answer this question:\n{question}"
        ),
    ]
)

# Optional: Split into chunks for better processing
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
docs_split = splitter.split_documents(documents)

# Turn the split documents into vectors
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs_split, embedding)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
)

while True:
    query = input("Ask a question (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = qa_chain({"question": query})
    print("\nAnswer:")
    print(result["answer"])
    print("-" * 40)
