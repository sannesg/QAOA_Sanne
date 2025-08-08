from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory

# ----- Helper imports -----
from agent_utils import (
    process_documents,
    load_context,
)
from pathlib import Path


class Explainer:
    def __init__(self, model="gpt-4.1", temperature=0):
        """
        Initialize the Explainer with the context for QAOA package components.

        Args:
            description (str): The description of the parts to explain.
            model (str): The language model to use.
            temperature (float): The temperature for the language model.
        """
        self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        self.context = ""
        self.set_total_context()  # Set the total context, a.k.a. the documentation strings
        self.query_context = ""  # This will hold the context for the specific query
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            input_key="description",
            return_messages=True,
            max_token_limit=1000,
        )

        self.prompt = PromptTemplate(
            input_variables=["description", "context", "chat_history"],
            template="""
        You are an expert on the QAOA package. You get a list over what the USER wants you to explain (they can be for example classes, methods, etc.) and you are going to explain how they work and what attributes, args, and returns they have.
        
        The parts you want to explain are: {description}
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
        self.chain = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory
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

    def get_selected_context(self, description):
        """Get the current context. NB: Needs to be called after set_context."""
        # Process the documents and create a vector store

        self.vectorstore = process_documents(self.context)

        # Create a retriever from the vector store
        self.retriever = self.vectorstore.as_retriever()

        # Get relevant documents based on the description
        docs = self.retriever.get_relevant_documents(description)

        # Join the page content of the documents to form the context in a string format
        self.query_context = "\n".join([doc.page_content for doc in docs])

    def explain(self, description):
        """Generate an explanation using the specified context chunk."""
        # Update the self.context with the relevant parts of the documentation strings based on the description
        self.get_selected_context(description)

        # Invoke the chain with the description and the updated context
        result = self.chain.invoke(
            {"description": description, "context": self.query_context}
        )
        return result.get("text", result)


"""
explainer = Explainer()
result = explainer.explain("QAOA class, Problems classes, Mixers classes, Initial states classes")
print(result)
"""
