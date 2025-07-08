from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings

# ----- Helper imports -----
from agent_utils import (
    extract_class_docstrings_from_string,
    load_context,
    process_documents,
)

# ----- Helper function imports -----
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
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            input_key="description",
            return_messages=True,
            max_token_limit=1000,
        )
        self.context = ""
        self.set_context()
        self.prompt = PromptTemplate(
            input_variables=["description", "context"],
            template="""
        You are an expert on the QAOA package. You get a list over what the USER wants you to explain (they can be for example classes, methods, etc.) and you are going to explain how they work and what attributes, args, and returns they have.
        
        The parts you want to explain are: {description}
        Your context is the documentation strings for the code: {context}
        
        Make it helpful so that the USER understand the overall meaning of the parts of the package and also how it is used in a code. 
        If you are explaining a method, include the class it belongs to. If you are explaining a class, include its methods and attributes there are any. If you are explaining a variable, include its type and purpose.
        Be concise and structured. 
        If the USER asks for a specific part of the QAOA package, make sure to explain that part in detail.
        Make subtitles for each part you explain, and do NOT use lists or numbered lists.
        Do not include anything the USER has not asked for.
        """,
        )
        self.chain = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory
        )  # added memory=self.memory

    def set_context(self):
        """Set or update the context variable with documentation."""
        folder_path = Path(r"C:\Users\sanne\QAOA_Sanne\qaoa")
        file_paths = [
            str(file)
            for file in folder_path.rglob("*")
            if file.is_file() and file.suffix in [".py", ".ipynb", ".txt", ".md"]
        ]

        self.context = [
            extract_class_docstrings_from_string(text)
            for text in load_context(file_paths)
        ]

        """
        repo_path = "./qaoa"
        docs_py = load_python_files(
            repo_path
        )  # switched from load_python_files to load_python_script
        extracted_docs_py = extract_class_docstrings_from_documents(docs_py)

        all_docstrings = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in extracted_docs_py
        ]

        chunks = []
        current = ""
        for doc in all_docstrings:
            if len(current) + len(doc) + 2 > max_chars:
                chunks.append(current)
                current = doc
            else:
                current += "\n\n" + doc
        if current:
            chunks.append(current)
        self.context_chunks = chunks  # Store as a list of chunks
        """

    def explain(self, description, chunk_index=0):
        """Generate an explanation using the specified context chunk."""
        context = self.context[chunk_index] if self.context else ""
        result = self.chain.invoke({"description": description, "context": context})
        return result.get("text", result)


"""
explainer = Explainer()
result = explainer.explain("QAOA class, Problems classes, Mixers classes, Initial states classes")
print(result)
"""
