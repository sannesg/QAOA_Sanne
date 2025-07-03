from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ----- Helper imports -----
from agent_utils import (
    extract_docstrings_from_documents,
    load_python_files,
    creating_vectorstore,
)


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
        You are an expert on the QAOA package. You are a list over what you want to explain (they can be for example classes, methods, etc.) and you are going to explain how they work and what attributes, args, and returns they have to the USER.
        
        The parts you want to explain are: {description}
        Your context is the documentation strings for the code: {context}
        
        Make it helpful so that the USER understand the overall meaning of the parts of the package and also how it is used in a code. Be concise and structured. 
        """,
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def set_context(self, max_chars=3000):
        """Set or update the context variable with documentation."""
        repo_path = "./qaoa"
        docs_py = load_python_files(repo_path)
        extracted_docs_py = extract_docstrings_from_documents(docs_py)
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

    def explain(self, chunk_index=0):
        """Generate an explanation using the specified context chunk."""
        context = self.context_chunks[chunk_index] if self.context_chunks else ""
        result = self.chain.invoke(
            {"description": self.description, "context": context}
        )
        return result.get("text", result)


explainer = Explainer(
    """Plan over which components of the QAOA package to explain:

1. QAOA class: This is the main class of the QAOA package. It is used to create an instance of the QAOA algorithm with a specific problem, mixer, and initial state.
2. Problems classes: These classes define the problem that the QAOA algorithm will solve. The valid problems are ExactCover, GraphProblem, MaxKCutOneHot, MaxKCutBinaryPowerOfTwo, MaxKCutBinaryFullH, and PortifolioOptimization.
3. Mixers classes: These classes define the mixer that the QAOA algorithm will use. The valid mixers are X, XY, Groover, MaxKCutGrover, and MaxKCutLX.
4. Initial states classes: These classes define the initial state that the QAOA algorithm will start from. The valid initial states are Dicke, Dicke1_2, Plus, LessThanK, and StateVector."""
)
explanation = explainer.explain()
print(explanation)
