# Load API key
from dotenv import load_dotenv
load_dotenv()

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

class CodeAssistant:
    def __init__(self, context_files: Optional[list[Union[str, Path]]]=None):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = [self.execute_code]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        if context_files:
            self._load_context(context_files)
        self.agent = self._initialize_agent()
        
    @tool
    def execute_code(self, code: str) -> str:
        """Executes the provided Python code and returns the result or error."""
        try:
            # Remove Markdown code fences if present
            code = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE)
            code = re.sub(r"```$", "", code.strip())

            exec_globals = {}
            exec(code.strip(), exec_globals)
            return str(exec_globals.get('result', 'Code executed successfully, but no result variable found.'))
        except Exception as e:
            import traceback
            return f"Error executing code:\n{traceback.format_exc()}"

    def _load_context(self, paths:List[str]) -> None:
        """Loads and processes documents from the specified paths."""
        documents = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                print(f"File not found - {path}. Skipping.")
                continue
            if path.suffix == ".ipynb":
                documents.extend(self._load_notebook(path))
            elif path.suffix == ".py":
                documents.extend(self._load_python_script(path))
            elif path.suffix in [".txt", ".md"]:
                documents.extend(self._load_text_file(path))
            else:
                print(f"Unsupported file type - {path.suffix}. Skipping.")
            
        print(f"Loaded {len(documents)} documents.")

        if documents:
            docstring_docs = self._extract_docstrings_from_documents(documents)
            self._process_docments(docstring_docs)
            
    def _load_notebook(self, path: Path) -> List[Document]:
        """Load Jupyter notebook content."""
        with open(path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
        
        content = []
        for cell in notebook["cells"]:
            if cell["cell_type"] in ["markdown", "code"]:
                cell_content = "\n".join(cell["source"])
                content.append(cell_content)
                print(f"Loaded cell content:\n{cell_content}\n{'-'*50}")
        
        return [Document(page_content="\n\n".join(content))]
    
    def _load_python_script(self, path: Path) -> List[Document]:
        """Load Python script content."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return [Document(page_content=content)]
    
    def _load_text_file(self, path: Path) -> List[Document]:
        """Load text or markdown file content."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return [Document(page_content=content)]

    def _extract_docstrings_from_documents(self, docs: List[Document]) -> List[Document]:
        docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
        extracted_docs = []

        for doc in docs:
            matches = re.findall(docstring_pattern, doc.page_content)
            for match in matches:
                extracted_docs.append(
                    Document(page_content=match.strip(), metadata=doc.metadata)
                )

        return extracted_docs
    
    def _process_docments(self, docs: List[Document]) -> None:
        """Process and store documents in vectorstore."""
        print(f"Processing {len(docs)} raw documents")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        print(f"Number of split documents: {len(split_docs)}")
        
        if not split_docs:
            print("No documents to process after splitting. Skipping vectorstore creation.")
            return

        print(f"Number of split documents: {len(split_docs)}")
        
        embedding = OpenAIEmbeddings()
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, embedding)
        else:
            self.vectorstore.add_documents(split_docs)
            
    def _initialize_agent(self) -> AgentExecutor:
        """Initialize and return the agent executor."""
        # Define prompts
        code_suggestion_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Python coding assistant. You suggest code based on the provided context.

Available Context:
{context}

Task:
1. Generate Python code to solve: {question}
2. The code should be complete and executable
3. Include a 'result' variable with the main output
4. Format the code in markdown with ```python code fences
5. If context is available, use relevant examples from it
"""
        )
        if not self.vectorstore:
            print("Vectorstore is not initialized. Retrieval-based QA will not work.")
            return None

        # Create retrieval chain for code suggestion
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}) if self.vectorstore else None,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": code_suggestion_prompt},
        )

        # Create agent for code testing/execution
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

context_files = ["./examples/MaxCut/KCutExamples.ipynb"]

assistant = CodeAssistant(context_files)

query = "Creaye a QAOA instance using the MaxKCutBinaryPowerOfTwo problem, X mixer and Plus initial state for k = 2, making a random graph."
result = assistant.qa_chain({"question": query})
print("\nAnswer:")
print(result["answer"])

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# rag_chain = RetrievalQA.from_chain_type(
#     llm=llm, retriever=retriever, chain_type="stuff"
# )

# # Instantiate the LLM
# llm = ChatOpenAI(model="gpt-4", temperature=0)

# # List of tools for the agent
# tools = [execute_code]

# # Define prompt
# code_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""You are a Python coding assistant.

# You are provided with the following context:
# {context}

# Your task is to answer the following question based on the context:
# {question}
# """
# #     template="""You are a Python code testing assistant.

# # You're given Python code. Your task is to **run** the code and report **any errors** that occur during execution. Use the `execute_code` tool to help with this.

# # Code:
# # {input}
# # """
# )

# notebook_path = "./examples/MaxCut/KCutExamples.ipynb"

# with open(notebook_path, "r", encoding="utf-8") as f:
#     notebook = json.load(f)

# context = []
# for cell in notebook["cells"]:
#     if cell["cell_type"] == "markdown":
#         context.append("\n".join(cell["source"]))
#     elif cell["cell_type"] == "code":
#         context.append("\n".join(cell["source"]))

# context_string = "\n\n".join(context)

# # Convert context_string into a list of Document objects
# documents = [Document(page_content=context_string)]

# splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
# context_split = splitter.split_documents(documents)

# # Turn the split documents into vectors
# embedding = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(context_split, embedding)

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     memory=memory,
#     combine_docs_chain_kwargs={"prompt": code_prompt},
# )

# query = "Create a QAOA instance using the MaxKCutBinaryPowerOfTwo problem, X mixer and Plus initial state for k = 2, making a random graph."

# result = qa_chain({"question": query})
# print("\nAnswer:")
# print(result["answer"])

# Sample code to test
# code_to_test = """
# import networkx as nx
# import numpy as np
# from qaoa.problems import MaxKCutBinaryPowerOfTwo

# G = nx.Graph()
# G.add_nodes_from(np.arange(0, 5, 1))  # Create a graph with 5 nodes
# G.add_weighted_edges_from(
#     [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (3, 4, 1.0), (4, 2, 1.0)]
# )

# problem = MaxKCutBinaryPowerOfTwo(G=G, k_cuts=2)
# result = problem  # Store result to be returned
# """

# # Initialize the agent
# code_agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# # Run the agent with the code to test
# answer = code_agent.run(f"Please run the following code and report any errors:\n\n{code_to_test}")

# # Output the result
# print(answer)


