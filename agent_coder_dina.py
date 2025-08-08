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
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)
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
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",  # See prompt template for usage
            input_key="question",
            return_messages=True,
            max_token_limit=1000,  # Limit memory size to avoid excessive context
        )
        self.vectorstore = None
        if context_files:
            self.context = agent_utils.load_context(context_files)
            self.vectorstore = agent_utils.process_documents(self.context)

        self._initialize_agent()

    # @tool We don't use this as a tool anymore to save on API calls
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

    # def _process_documents(self, paths: List[str]) -> None:
    #     """Process and store documents in vectorstore."""
    #     docs_str = agent_utils.load_context(paths)

    #     print(f"Processing {len(docs_str)} raw documents.")
    #     try:
    #         docs = [Document(page_content=doc.strip()) for doc in docs_str]

    #         splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    #         split_docs = splitter.split_documents(docs)
    #         print(f"Generated {len(split_docs)} split documents.")

    #         if not split_docs:
    #             print("No documents to process after splitting. Skipping vectorstore creation.")
    #             return

    #         embedding = OpenAIEmbeddings()

    #         self.vectorstore = FAISS.from_documents(split_docs, embedding)
    #         print(f"Created vectorstore with {len(split_docs)} documents.")
    #     except Exception as e:
    #         print(f"Error processing documents: {str(e)}")
    #         self.vectorstore = None

    def _initialize_agent(
        self, context_files: Optional[list[Union[str, Path]]] = None
    ) -> None:
        """Initialize and return the agent executor."""
        # Define prompts
        if context_files:
            self.context = agent_utils.load_context(context_files)

        code_suggestion_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a AI, a Python coding assistant. 
            
You have four tasks based on the input:

1. If asked a query with no error information, generate Python code to solve the task.
2. If provided with an error message, analyze the error and suggest improvements to the code without generating new code.
3. If asked to improve code based on suggestions, generate improved Python code considering the provided feedback.
4. If an explaination is requested, provide a concise explanation.

Context:
{context}

Conversation history:
{chat_history}

Human: {question}
AI:

Guidelines:
1. Generate Python code to solve the task provided by the Human.
2. The code should be complete and executable
3. The code should include all necessary imports and mainly use the QAOA package. Don't include any imports that are not used.
4. The code should be formatted in markdown with ```python code fences
5. The code should include BRIEF comments in the code explaining key steps.
6. If analyzing an error, provide concise suggestions for improvement without generating code.
7. If analyzing an error and the error is 'NoneType', the function probably updates an internal variable, rather than returning a value.
In this case, try to find the variable that is updated and suggest using this instead.
8. Phrase all responses as if it is the first response to its corresponding query. i.e. don't mention executing code or changes made after executing code. 
9. After generating code, briefly explain the approach and any potential limitations in the code or discrepancies between the code and the task.
10. For parts of the task that are unspecified, provide brief reasoning for your choices.
11. Refer to previous tasks and responses in the conversation to maintain context and continuity.
""",
        )
        # Initialize chain that handles memory
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

    # Main function
    def generate_and_test_code(self, query: str, max_iterations: int = 3) -> None:
        """Generate code, test it, and improve based on feedback."""
        current_code = None
        error_analysis = None
        # last_error = None

        for iteration in range(max_iterations):
            print(f"\033[90m\n--- Iteration {iteration + 1} ---\033[0m")

            # Generate new or improve old code
            if current_code is None:
                print("\033[90m\nGenerating initial response...\033[0m")
                result = self.qa_chain.invoke({"question": query})
            else:
                print("\033[90m\nImproving response based on previous error...\033[0m")
                result = self.qa_chain.invoke(
                    {
                        "question": f"Improve this code based on the following feedback: {error_analysis}\nOriginal task: {query}\nCode:\n{self._extract_code_block(current_code)}"
                    }
                )

            current_code = result["answer"]  # Extract response text

            print("\033[90m\nResponse:\033[0m")
            print(f"\033[90m\n{current_code}\033[0m")

            # Execute the code if it contains a code block
            if "```" in current_code:
                matplotlib.use(
                    "Agg"
                )  # Use a non-interactive backend for matplotlib (no verbose output in console)
                print("\033[96m\nExecuting code...\033[0m")
                execution_result = self.execute_code(
                    self._extract_code_block(current_code)
                )
                matplotlib.use("TkAgg")  # Reset to default backend
                print(f"\033[96m{execution_result}\033[0m")
                if "ERROR" in execution_result:
                    print(f"\033[96m\nGenerating error analysis...\033[0m")
                    result = self.qa_chain.invoke(
                        {
                            "question": f"""Analyze the following error: {execution_result}. 
                        Provide suggestions for improving the code without generating new code."""
                        }
                    )
                    error_analysis = result["answer"]
                    print(f"\033[96mError analysis: {error_analysis}\033[0m")
                else:
                    return current_code  # Return the response if no errors occurred

            else:
                return current_code

        print(f"\nReached maximum iterations ({max_iterations})")
        return current_code  # Return the last generated response when max tries are reached

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown block."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


if __name__ == "__main__":

    # Example usage
    context_files = ["./examples/MaxCut/KCutExamples.ipynb", "./qaoa/qaoa.py"]
    assistant = CodeAssistant(context_files)

    # query = "Create a random connected graph with 10 nodes. Include visualization."
    # query = "Create a qaoa instance using onehot encoding."

    # final_code = assistant.generate_and_test_code(query)
    # print("\nFinal response:")
    # print(final_code)

    # First query
    query1 = "Create a qaoa instance using onehot encoding."
    print("\nFirst query: ")
    print(f"\033[1m{query1}\033[0m")
    final_code1 = assistant.generate_and_test_code(query1)
    print("\nFinal response to first query:")
    print(f"\033[1m{final_code1}\033[0m")

    # Second query relies on memory of the first one
    query2 = "Why did you choose the initial state and mixer like that?"
    print("\nSecond query: ")
    print(f"\033[1m{query2}\033[0m")
    final_response2 = assistant.qa_chain.invoke({"question": query2})
    print("\nFinal response to second query:")
    print(f"\033[1m{final_response2["answer"]}\033[0m")

    # Third query
    query3 = """Create a qaoa circuit using a random 10-node connected graph for k = 3 using binary encoding and the full hamiltonian.
    Visualize both the graph and the circuit."""
    print("\nThird query: ")
    print(f"\033[1m{query3}\033[0m")
    final_code3 = assistant.generate_and_test_code(query3)
    print("\nFinal response to third query:")
    print(f"\033[1m{final_code3}\033[0m")

    # # Print memory summary
    # print("\033[95m\nMemory summary:\033[0m")
    # print(f"\033[95m{assistant.memory.buffer}\033[0m")
