# to make the code splitter
import re
from langchain_core.documents import Document


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
