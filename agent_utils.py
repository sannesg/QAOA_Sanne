# to make the code splitter
import re
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import ast
from langchain.schema import Document


# ----- Making the Code Splitter -----
def extract_docstrings_from_documents(docs):
    docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
    extracted_docs = []

    for doc in docs:
        matches = re.findall(docstring_pattern, doc.page_content)
        for match in matches:
            extracted_docs.append(
                Document(page_content=match.strip(), metadata=doc.metadata)
            )

    return extracted_docs


# ----- Extracting class docstrings from documents -----
def extract_class_docstrings_from_documents(docs):
    extracted_docs = []

    for doc in docs:
        try:
            tree = ast.parse(doc.page_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        extracted_docs.append(
                            Document(
                                page_content=docstring.strip(), metadata=doc.metadata
                            )
                        )
        except SyntaxError:  # Skip documents that can't be parsed
            continue

    return extracted_docs


# ----- To load the files from the folder ------
def load_python_files(repo_path):
    """
    Load all Python files from the specified repository path.
    """
    loader_py = DirectoryLoader(repo_path, glob="**/*.py")
    docs_py = loader_py.load()
    return docs_py


def load_text_files(repo_path):
    """
    Load all text files from the specified repository path.
    """
    loader_txt = DirectoryLoader(repo_path, glob="**/*.txt")
    docs_txt = loader_txt.load()
    return docs_txt


# ----- To create a vector index of the split elements -----
def creating_vectorstore(docs):
    """
    Create a vector store from the provided documents.
    """
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore
