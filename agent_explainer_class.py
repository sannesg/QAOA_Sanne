from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ----- Helper imports -----
from agent_utils import (
    extract_docstrings_from_documents,
    load_python_files,
    creating_vectorstore,
)

# ----- To test Explainer with Planner -----
from agent_planner_class_sanne import result


class Explainer:
    def __init__(self, description, model="gpt-4", temperature=0):
        """
        Initialize the Explainer with the context for QAOA package components.

        Args:
            description (str): The description of the parts to explain.
            model (str): The language model to use.
            temperature (float): The temperature for the language model.
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.context = ""
        self.set_context()
        self.description = description
        self.prompt = PromptTemplate(
            input_variables=["description", "context"],
            template="""
        You are an expert on the QAOA package. You get a list over what you want to explain (they can be for example classes, methods, etc.) and you are going to explain how they work and what attributes, args, and returns they have.
        
        The parts you want to explain are: {description}
        Your context is the documentation strings for the code: {context}
        
        Make it helpful so that the USER understand the overall meaning of the parts of the package and also how it is used in a code. 
        If you are explaining a method, include the class it belongs to. If you are explaining a class, include its methods and attributes. If you are explaining a variable, include its type and purpose.
        Be concise and structured. 
        Do not include anything the USER has not asked for.
        """,
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def set_context(self, max_chars=3000):
        """Set or update the context variable with documentation."""
        repo_path = "./qaoa"
        docs_py = load_python_files(repo_path)
        extracted_docs_py = extract_docstrings_from_documents(docs_py)

        # only keep docstrings that contain specific keywords
        keywords = ["Args", "Attributes", "Returns", "Raises", "Example", "Class"]
        all_docstrings = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in extracted_docs_py
            if any(
                keyword
                in (doc.page_content if hasattr(doc, "page_content") else str(doc))
                for keyword in keywords
            )
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

    def explain(self, chunk_index=0):
        """Generate an explanation using the specified context chunk."""
        context = self.context_chunks[chunk_index] if self.context_chunks else ""
        result = self.chain.invoke(
            {"description": self.description, "context": context}
        )
        return result.get("text", result)


explainer = Explainer(result)
explanation = explainer.explain()
print("\n\nExplanation:\n\n", explanation)
