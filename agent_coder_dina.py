# Load API key
from dotenv import load_dotenv

load_dotenv()

import agent_utils

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import traceback
import matplotlib

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain

from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.document_loaders import DirectoryLoader


class CodeAssistant:
    def __init__(self, context_files: Optional[list[Union[str, Path]]] = None):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = [self.execute_code]
        self.memory = ConversationSummaryMemory(
            llm=self.llm,  # you must pass an LLM to generate summaries
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
        )
        self.vectorstore = None
        if context_files:
            self._process_documents(context_files)

        self._initialize_agent()

    # @tool
    def execute_code(self, code: str) -> str:
        """Executes the provided Python code and returns only error messages if any occur."""
        try:
            # Remove Markdown code fences if present
            code = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE)
            code = re.sub(r"```$", "", code.strip())

            # Redirect stdout to suppress circuit diagrams

            exec_globals = {}
            f = io.StringIO()

            with redirect_stdout(f), redirect_stderr(f):
                exec(code.strip(), exec_globals)

            # Only return success message if no errors
            return "SUCCESS: Code executed without errors"

        except Exception as e:
            # Return just the error type and message, not full traceback
            return f"ERROR: {type(e).__name__}: {str(e)}"

    def _process_documents(self, paths: List[str]) -> None:
        """Process and store documents in vectorstore."""
        docs_str = agent_utils.load_context(paths)

        print(f"Processing {len(docs_str)} raw documents")
        try:
            docs = [Document(page_content=doc.strip()) for doc in docs_str]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200
            )
            split_docs = splitter.split_documents(docs)
            print(f"Number of split documents: {len(split_docs)}.")

            if not split_docs:
                print(
                    "No documents to process after splitting. Skipping vectorstore creation."
                )
                return

            embedding = OpenAIEmbeddings()

            self.vectorstore = FAISS.from_documents(split_docs, embedding)
            print(f"Created vectorstore with {len(split_docs)} documents.")
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            self.vectorstore = None

    def _initialize_agent(
        self, context_files: Optional[list[Union[str, Path]]] = None
    ) -> None:
        """Initialize and return the agent executor."""
        # Define prompts
        if context_files:
            self.context = agent_utils.load_context(context_files)

        code_suggestion_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Python coding assistant. 
            
You have three tasks based on the input:

1. If asked a query with no error information, generate Python code to solve the task.
2. If provided with an error message, analyze the error and suggest improvements to the code without generating new code.
3. If asked to improve code based on suggestions, generate improved Python code considering the provided feedback.

Context:
{context}

Chat History:
{chat_history}

Task:
{question}

Guidelines:
1. Generate Python code to solve the task
2. The code should be complete and executable
3. The code should include all necessary imports and mainly use the QAOA package
4. The code should be formatted in markdown with ```python code fences
5. The code should include BRIEF comments in the code explaining key steps.
6. If analyzing an error, provide concise suggestions for improvement without generating code.
7. Phrase all responses as if it is the first response to the user.
8. After generating code, briefly explain the approach and any potential limitations in the code or discrepancies between the code and the task.
9. For parts of the task that are unspecified, provide brief reasoning for your choices.
""",
        )
        # self.qa_chain = RetrievalQA.from_chain_type(
        #     llm = self.llm,
        #     chain_type="stuff",
        #     retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}) if self.vectorstore else None,
        #     chain_type_kwargs={
        #         "prompt": code_suggestion_prompt,
        #         # "document_variable_name": "context"
        #     },
        #     input_key="question",
        #     output_key="answer",
        #     return_source_documents=True
        #     )

        # Using memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=(
                self.vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
                if self.vectorstore
                else None
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": code_suggestion_prompt,
            },
        )

    def generate_and_test_code(self, query: str, max_iterations: int = 3) -> None:
        """Generate code, test it, and improve based on feedback."""
        current_code = None
        error_analysis = None
        # last_error = None

        for iteration in range(max_iterations):
            print(f"\n===== Iteration {iteration + 1} =====")

            # Generate or improve code
            if current_code is None:
                print("Generating initial response...")
                result = self.qa_chain.invoke(
                    {
                        # "context": self.context,
                        "question": query
                    }
                )
            else:
                print("Improving response based on previous error...")
                result = self.qa_chain.invoke(
                    {
                        # "context": self.context,
                        "question": f"Improve this code based on the following feedback: {error_analysis}\nOriginal task: {query}\nCode:\n{self._extract_code_block(current_code)}"
                    }
                )

            current_code = result["answer"]

            print("\nResponse:")
            print(current_code)

            if "```" in current_code:
                matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
                execution_result = self.execute_code(
                    self._extract_code_block(current_code)
                )
                matplotlib.use("TkAgg")  # Reset to default backend
                print("\nExecution Result:")
                print(execution_result)
                if "ERROR" in execution_result:
                    result = self.qa_chain.invoke(
                        {
                            # "context": self.context,
                            "question": f"""Analyze the following error: {execution_result}. 
                        Provide suggestions for imporving the code without generating new code."""
                        }
                    )
                    error_analysis = result["answer"]
                    print("\nError Analysis:")
                    print(error_analysis)
                else:
                    return current_code

            else:
                return current_code

        print(f"\nReached maximum iterations ({max_iterations})")
        # return self._extract_code_block(current_code)
        return current_code

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown block."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


if __name__ == "__main__":
    # Example usage
    context_files = ["./examples/MaxCut/KCutExamples.ipynb"]
    assistant = CodeAssistant(context_files)

    # # query = "Create a random connected graph with 10 nodes. Include visualization."
    # query = "Create a qaoa instance using onehot encoding."

    # final_code = assistant.generate_and_test_code(query)
    # print("\nFinal response:")
    # print(final_code)

    # First query
    query1 = "Create a qaoa instance using onehot encoding."
    print("\n--- First Query ---")
    final_code1 = assistant.generate_and_test_code(query1)
    print("\nFinal response to first query:")
    print(final_code1)

    # Second query relies on memory of the first one
    query2 = "Can you explain why you used onehot encoding?"
    print("\n--- Second Query ---")
    final_response2 = assistant.qa_chain.invoke({"question": query2})
    print("\nFinal response to second query:")
    print(final_response2["answer"])

    print("\n--- Memory Summary ---")
    print(assistant.memory.buffer)
