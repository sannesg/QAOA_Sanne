from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory

# ----- Helper imports -----
from agent_utils import (
    process_documents,
    load_context,
)
from pathlib import Path


class Explainer:
    def __init__(self, model="gpt-4", temperature=0):
        """
        Initialize the Explainer with the context for QAOA package components.

        Args:
            description (str): The description of the parts to explain.
            model (str): The language model to use.
            temperature (float): The temperature for the language model.
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.context = ""
        self.set_total_context()  # Set the total context, a.k.a. the documentation strings
        self.vectorstore = process_documents(self.context)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
            max_token_limit=1000,
        )

        self.prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""
        You are an expert on the QAOA package. You get a list over what the USER wants you to explain (they can be for example classes, methods, etc.) and you are going to explain how they work and what attributes, args, and returns they have.
        
        The parts you want to explain are: {question}
        Your context is the documentation strings for the code: {context}
        Here is the chat history: {chat_history}
        
        Make it helpful so that the USER understand the overall meaning of the parts of the package and also how it is used in a code. 
        If you are explaining a method, include the class it belongs to. If you are explaining a class, include its methods and attributes there are any. If you are explaining a variable, include its type and purpose.
        Be concise and structured. 
        If the USER asks for a specific part of the QAOA package, make sure to explain that part in detail.
        Make subtitles for each part you explain, and do NOT use lists or numbered lists.
        NEVER include code snippets in your response.
        Do not include anything the USER has not asked for.
        """,
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            retriever=(
                self.vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
            ),
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )  # added memory=self.memory

    def set_total_context(self):
        """Set or update the context variable with documentation."""
        folder_path = Path(r"C:\Users\sanne\QAOA_Sanne\qaoa")

        # Only get .py files for docstring extraction
        py_file_paths = [
            str(file) for file in folder_path.rglob("*.py") if file.is_file()
        ]

        # Load only .py content
        self.context = load_context(py_file_paths)

    def explain(self, question):
        """Generate an explanation using the specified context chunk."""
        result = self.chain.invoke({"question": question, "context": self.context})
        return result.get("answer", result)


"""
explainer = Explainer()
result = explainer.explain("QAOA class, Problems classes, Mixers classes, Initial states classes")
print(result)
"""
