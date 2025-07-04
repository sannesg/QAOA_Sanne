# Load API key
from dotenv import load_dotenv
load_dotenv()

from agent_utils import *

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import io
import sys
from contextlib import redirect_stdout
import traceback

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
# from langchain_community.document_loaders import DirectoryLoader

class CodeAssistant:
    def __init__(self, context_files: Optional[list[Union[str, Path]]]=None):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = [self.execute_code]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        if context_files:
            self._load_context(context_files)
            
        self._initialize_agent()
        
    # @tool
    def execute_code(code: str) -> str:
        """Executes the provided Python code and returns only error messages if any occur."""
        try:
            # Remove Markdown code fences if present
            code = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE)
            code = re.sub(r"```$", "", code.strip())

            # Redirect stdout to suppress circuit diagrams

            exec_globals = {}
            f = io.StringIO()
            
            with redirect_stdout(f):
                exec(code.strip(), exec_globals)
                
            # Only return success message if no errors
            return "SUCCESS: Code executed without errors"
            
        except Exception as e:
            # Return just the error type and message, not full traceback
            return f"ERROR: {type(e).__name__}: {str(e)}"

    def _load_context(self, paths:List[str]) -> None:
        """Loads and processes documents from the specified paths."""
        documents = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                print(f"File not found - {path}. Skipping.")
                continue
            if path.suffix == ".ipynb":
                documents.append(self._load_notebook(path))
            elif path.suffix in [".txt", ".md"]:
                documents.append(self._load_text_file(path))
            elif path.suffix == ".py":
                py_str = self._load_python_script(path)
                docstring_str = self._extract_docstrings_from_documents(py_str)
                documents.append(docstring_str)
            else:
                print(f"Unsupported file type - {path.suffix}. Skipping.")
            
        print(f"Loaded {len(documents)} documents.")
        return documents
        # if documents:
        #     self._process_documents(documents)
            
    def _load_notebook(self, path: Path) -> str:
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
        # return [Document(page_content="\n\n".join(content))]
        
    
    def _load_python_script(self, path: Path) -> str:
        """Load Python script content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        # return [Document(page_content=content)]
    
    def _load_text_file(self, path: Path) -> str:
        """Load text or markdown file content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        # return [Document(page_content=content)]

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
    
    def _process_documents(self, docs: List[Document]) -> None:
        """Process and store documents in vectorstore."""
        print(f"Processing {len(docs)} raw documents")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        print(f"Number of split documents: {len(split_docs)}")
        
        if not split_docs:
            print("No documents to process after splitting. Skipping vectorstore creation.")
            return
        
        embedding = OpenAIEmbeddings()
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, embedding)
        else:
            self.vectorstore.add_documents(split_docs)
            
    def _initialize_agent(self) -> None:
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
3. Include all necessary imports
4. Format the code in markdown with ```python code fences
5. If context is available, use relevant examples from it

Execution Guidelines:
- The code will be automatically executed after generation
- If the execution returns "SUCCESS:" it means the code worked
- If the execution returns "ERROR:" it means the code failed
- Your response should ONLY contain the code block with no additional commentary
- Don't comment on it if the code doesn't return anything
"""
)
        # if not self.vectorstore:
        #     print("Vectorstore is not initialized. Retrieval-based QA will not work.")

        # Create retrieval chain for code suggestion
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}) if self.vectorstore else None,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": code_suggestion_prompt},
        )

        # # Create agent for code testing/execution
        # self.agent = initialize_agent(
        #     tools=self.tools,
        #     llm=self.llm,
        #     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        #     verbose=True,
        #     return_intermediate_steps=True,
        #     handle_parsing_errors=True,
        #     max_iterations=3,
        #     agent_kwargs={
        #         'prefix': """You are a coding assistant that executes Python code. When testing code:
        # 1. Always use the execute_code tool
        # 2. If output starts with "SUCCESS", the code worked
        # 3. If output starts with "ERROR", analyze the error type and message
        # 4. Never show circuit diagrams or verbose output"""
        #     }
        # )
    
    def generate_and_test_code(self, query: str, max_iterations: int = 3) -> None:
        """Generate code, test it, and improve based on feedback."""
        current_code = None
        last_error = None
        
        for iteration in range(max_iterations):
            print(f"\n===== Iteration {iteration + 1} =====")
            
            # Generate or improve code
            if current_code is None:
                print("Generating initial code...")
                result = self.qa_chain.invoke({"question": query})
            else:
                print("Improving code based on last error...")
                result = self.qa_chain.invoke({
                    "question": f"Fix this code that failed with error: {error_analysis}\nOriginal task: {query}\nCode:\n{current_code}"
                }) 
                
            current_code = result["answer"]
            
            print("\nGenerated Code:")
            print(current_code)
            
            # Test code execution
            # execution_result = self.agent.invoke({
            #     "input": f"Execute and validate this code and report any errors:\n{current_code}"
            # })
            
            if "```python" in current_code:
                execution_result = self.execute_code(current_code)
                print("\nExecution Result:")
                print(execution_result)
                if "ERROR" in execution_result:
                    result = self.qa_chain.invoke(f"The code {self._extract_code_block(current_code)} has the following error: {execution_result}. Make a comment on the error and what improvements should be made to the code.")
                    error_analysis = result["answer"]
                    print("\nError Analysis:")
                    print(error_analysis)
                else:
                    return current_code
            
            # last_error = execution_result["output"]

            # print("\nExecution result:")
            # print(last_error)

            # steps = execution_result.get("intermediate_steps", [])
            # for action, observation in steps:
            #     if isinstance(observation, str) and "SUCCESS" in observation:
            #         print("Code executed successfully!")
            #         return self._extract_code_block(current_code)
        
        print(f"\nReached maximum iterations ({max_iterations})")
        # return self._extract_code_block(current_code)
        return current_code

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown block."""
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

# Example usage
context_files = ["./examples/MaxCut/KCutExamples.ipynb", "./examples/MaxCut/ToyExample.ipynb"]
assistant = CodeAssistant(context_files)

query = "Create a qaoa instance usinga random graph with 8 nodes for k = 2."

final_code = assistant.generate_and_test_code(query)
print("\nFinal Code:")
print(final_code)