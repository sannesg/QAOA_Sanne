# to make the code splitter
import re
from pathlib import Path
from typing import List
import json

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import ast
from langchain.schema import Document

# def extract_docstrings_from_documents(docs):
#     docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
#     extracted_docs = []

#     for doc in docs:
#         matches = re.findall(docstring_pattern, doc.page_content)
#         for match in matches:
#             extracted_docs.append(
#                 Document(page_content=match.strip(), metadata=doc.metadata)
#             )

#     return extracted_docs


# def extract_class_docstrings_from_documents(docs:List[Document]) -> str:
#     extracted_docs = []

#     for doc in docs:
#         try:
#             tree = ast.parse(doc.page_content)
#             for node in ast.walk(tree):
#                 if isinstance(node, ast.ClassDef):
#                     docstring = ast.get_docstring(node)
#                     if docstring:
#                         extracted_docs.append(
#                             page_content=docstring.strip(), metadata=doc.metadata
#                         )
#         except SyntaxError:  # Skip documents that can't be parsed
#             continue
#     return extracted_docs

# # ----- To load the files from the folder ------
def load_python_files(repo_path):
    """
    Load all Python files from the specified repository path.
    """
    loader_py = DirectoryLoader(repo_path, glob="**/*.py")
    docs_py = loader_py.load()
    return docs_py


# def load_text_files(repo_path):
#     """
#     Load all text files from the specified repository path.
#     """
#     loader_txt = DirectoryLoader(repo_path, glob="**/*.txt")
#     docs_txt = loader_txt.load()
#     return docs_txt


# # ----- To create a vector index of the split elements -----
# def creating_vectorstore(docs):
#     """
#     Create a vector store from the provided documents.
#     """
#     embedding = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(docs, embedding)
#     return vectorstore

# -----

def load_notebook(path: Path) -> str:
    """Load Jupyter notebook content."""
    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    content = []
    for cell in notebook["cells"]:
        if cell["cell_type"] in ["markdown", "code"]:
            cell_content = "\n".join(cell["source"])
            content.append(cell_content)
            # print(f"Loaded cell content:\n{cell_content}\n{'-'*50}")
    return "\n\n".join(content)

def load_python_script(path: Path) -> str:
    """Load Python script content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_text_file(path: Path) -> str:
    """Load text or markdown file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_class_docstrings_from_string(code: str) -> str:
    """Extract class-level docstrings from a Python source string."""
    extracted_docs = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    extracted_docs.append(docstring.strip())
    except SyntaxError:
        pass  # Skip files that fail to parse

    return "\n\n".join(extracted_docs)

def load_context(paths: List[str]) -> None:
    """Loads and processes documents from the specified paths."""
    context = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"File not found - {path}. Skipping.")
            continue
        if path.suffix == ".ipynb":
            context.append(load_notebook(path))
        elif path.suffix in [".txt", ".md"]:
            context.append(load_text_file(path))
        elif path.suffix == ".py":
            py_str = load_python_script(path)
            docstring_str = extract_class_docstrings_from_string(py_str)
            context.append(docstring_str)
        else:
            print(f"Unsupported file type - {path.suffix}. Skipping.")

    print(f"\nLoaded {len(context)} files.")
    return context

def process_documents(context_strs: List[str]) -> FAISS | None:
    """Process and store documents in vectorstore."""
    
    print(f"Processing {len(context_strs)} raw documents.")
    try:
        docs = [Document(page_content=context_str.strip()) for context_str in context_strs]
    
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        print(f"Generated {len(split_docs)} split documents.")
    
        if not split_docs:
            print("No documents to process after splitting. Skipping vectorstore creation.")
            return
    
        embedding = OpenAIEmbeddings()

        vectorstore = FAISS.from_documents(split_docs, embedding)
        print(f"Created vectorstore with {len(split_docs)} documents.")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        vectorstore = None
    return vectorstore