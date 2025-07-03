# Loading API key
from dotenv import load_dotenv

load_dotenv()

# To use the OpenAI LLM
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import (
    initialize_agent,
    AgentType,
    create_react_agent,
    AgentExecutor,
    Tool,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

# to make the code splitter
import re
from langchain_core.documents import Document

# To load the files from the folder
from langchain_community.document_loaders import DirectoryLoader

# To create a vector index of the split elements
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


# ----- Define placeholder tools (for future use) -----
@tool
def choose_initial_state(description: str) -> str:
    """Chooses initial state based on the description."""
    return "Initial state chosen based on description."


@tool
def choose_problem(description: str) -> str:
    """Chooses problem based on the description."""
    return "Problem chosen based on description."


@tool
def choose_mixer(description: str) -> str:
    """Chooses mixer based on the description."""
    return "Mixer chosen based on description."


tools = [choose_initial_state, choose_problem, choose_mixer]

# ----- LLM -----
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ----- Custom Prompt for the Planner Chain and the Planner Chain-----
planner_prompt = PromptTemplate(
    input_variables=["description"],
    template="""
You are an expert in the QAOA Python package and a code assistant to the USER. Your task is to create a plan (a prompt) for another AGENT to follow.

Given this USER's input: "{description}", do ONLY 1 of the following 2 options based on the USER's request:

1. If the USER requests specific code or how to implement QAOA, generate a numbered list of concise, implementation-focused steps to create a Python script using only the QAOA package.  
 - The title above the steps are ALWAYS "Plan to generate code using the QAOA package".
 - Do NOT write any code or call any tools.  
 - Only describe how to do each step.  
 - If the USER specifies initial state, problem, or mixer, add the type of initial state/problem/mixer in the relevant step in the template below. Otherwise, add the type:  
  Problem = MaxKCutBinaryPowerOfTwo, Initial state = Plus, Mixer = X.

    The step template you will use is:
        1. Import the necessary libraries such as qaoa, matplotlib, numpy, and networkx.
        2. Make a graph using networkx that represents the problem. Default to a complete graph with 4 nodes if not specified.
        3. Choose the problem.
        4. Choose the mixer.  
        5. Choose the initial state. 
        6. Create an instance of the QAOA class with the chosen initial state, problem, and mixer.
        7. Create parametrized circuits using the QAOA instance. Default to depth=1 if not specified.
        8. Draw graph using networkx.
        9. Draw quantum circuit using qaoa.draw_circuit().
        10. Use qaoa.sample_cost_landscape() to sample the cost landscape. Default parameters: angles= gamma: [-np.pi / 2, np.pi / 2, 15], beta: [1, 1 + 2 * np.pi, 15] in a library if not specified.
        11. Plot the cost landscape using plot_E(qaoa).
        12. Use qaoa.get_Exp(depth=1) to get the expectation value of the Hamiltonian.
 
 - IF the USER ONLY asks for a initialization type of code, then only include the steps 1-7.
 - IF the USER asks for a visualization, include spteps 8-9.
 - IF the USER asks for a cost landscape, include steps 10-12.

2. If the USER asks for an explanation about the QAOA package, generate a concise ONE-TWO-WORD-per-point list of components. 
 - Title of the list is ALWAYS "Plan over which components of the QAOA package to explain".
 - Use this case if: "explain", "explanation", "components", "parts", or "structure" is in the description and "code" or "implementation" is not.

   Template for the components:
        1. QAOA class
        2. Problems
        3. Mixers
        4. Initial states
 - IF the USER ONLY asks for a specific component, then only include that component in the list.

Strictly follow these rules:  
 - Output ONLY the numbered list of steps or the list of components.  
 - No additional commentary, no code, no explanations, no tool calls.  
 - If you cannot follow these rules, respond with: "Cannot provide a plan based on the input."

Remember: Be concise, focused, and precise.
""",
)

planner_chain = LLMChain(llm=llm, prompt=planner_prompt)

# -----Making the Code Agent-----
code_agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


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


"""
# "." because the files are in the same folder
repo_path = "./qaoa"

# Load all .py files
loader_py = DirectoryLoader(repo_path, glob="**/*.py")
docs_py = loader_py.load()

docstring_docs = extract_docstrings_from_documents(docs_py)

# making an embedding and vectorstore
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docstring_docs, embedding)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

rag_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff"
)
"""
# ----- Main loop -----

while True:
    query = input("Ask a question (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # plan = planner_chain.run({"description": query})
    plan = planner_chain.run(query)
    print("\nPlan:")
    print(plan)
    print("-" * 40)


"""
# Tools for the Planner Agent
@tool
def choose_initial_state(description: str) -> str:
"""
# Chooses initial state based on the description.
"""
@tool
def choose_problem(description: str) -> str:
"""
# Chooses problem based on the description.
"""
@tool
def choose_mixer(description: str) -> str:
"""
# Chooses mixer based on the description.
"""
    
# Use the LLM from OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Make list of tools for the Planner Agent
tools = [choose_initial_state, choose_problem, choose_mixer]

#Make prompt for the Planner Agent
planner_prompt = PromptTemplate(input_variables = ["description"], template = 
        "You are an expert in the QAOA python package and a code expert. You will only use the QAOA package to solve the problem as far as that is possible. " \
        "Given this task description: {description}, generate a numbered list of steps to generate a code using the QAOA package. " \
        "Each step should be one sentence. Keep it implementation-focused." \
        "The tools generates the code using the QAOA package, however you should not use the tools directly, but rather create a plan for how to use the tools. "\
        "The tools are: {tools}. If the user asks for a specific code, you can respond with the plan for how to generate the code.")

planner_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, prompt=planner_prompt)

# Main loop for interactive Q&A, does not contain memory yet
while True:
    query = input("Ask a question (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    plan = planner_agent.run(query)
    print("\nAnswer:")
    print(plan)
    print("-" * 40)
"""


# prompts
"""
You are an expert in the QAOA Python package and a code assistant to the USER. Your task is to create a plan (a prompt) for another AGENT to follow.

Here is some additional context:
{context}

It contains three lists: 1. Initial states, 2. Mixers, and 3. Problems. Each list contains valid options for the QAOA package.

Given this USER's input: "{description}", do ONLY 1 of the following 2 options based on the USER's request:

1. If the USER requests specific code or how to implement QAOA, generate a numbered list of concise, implementation-focused steps to create a Python script using only the QAOA package.  
 - The title above the steps are ALWAYS "Plan to generate code using the QAOA package".
 - Do NOT write any code or call any tools.  
 - Only describe how to do each step.  
 - The context contains exact names of valid initial states, mixers, and problems.
 - If any USER-specified initial state, problem, or mixer is NOT an exact match with an option in {context}, then:
     - Respond ONLY with: "The [initial state/problem/mixer] '[name]' is not a valid option based on the documentation."
     - Do NOT generate any steps.

 - Else:
    Generate the numbered steps using either the user-specified or default components (Default for each if missing: problem = MaxkCutPowerofTwo, mixer = X, initial state = Plus).
 - Always produce a numbered plan if no invalid options are found.

    The step template you will use otherwise is:
        1. Import the necessary libraries such as qaoa, matplotlib, numpy, and networkx.
        2. Make a graph using networkx that represents the problem. Default to a complete graph with 4 nodes if not specified.
        3. Choose the problem.
        4. Choose the mixer.
        5. Choose the initial state. 
        6. Create an instance of the QAOA class with the chosen initial state, problem, and mixer.
        7. Create parametrized circuits using the QAOA instance. Default to depth=1 if not specified.
        8. Draw graph using networkx.
        9. Draw quantum circuit using qaoa.draw_circuit().
        10. Use qaoa.sample_cost_landscape() to sample the cost landscape. Default parameters: angles= gamma: [-np.pi / 2, np.pi / 2, 15], beta: [1, 1 + 2 * np.pi, 15] in a library if not specified.
        11. Plot the cost landscape using plot_E(qaoa).
        12. Use qaoa.get_Exp(depth=1) to get the expectation value of the Hamiltonian.
 
 - IF the USER ONLY asks for a initialization type of code, then only include the steps 1-7.
 - IF the USER asks for a visualization, include spteps 8-9.
 - IF the USER asks for a cost landscape, include steps 10-12.

2. If the USER asks for an explanation about the QAOA package, generate a concise ONE-TWO-WORD-per-point list of components. 
 - Title of the list is ALWAYS "Plan over which components of the QAOA package to explain".
 - Use this case if: "explain", "explanation", "components", "parts", or "structure" is in the description and "code" or "implementation" is not.

   Template for the components:
        1. QAOA class
        2. Problems
        3. Mixers
        4. Initial states
 - IF the USER ONLY asks for a specific component, then only include that component in the list.

Strictly follow these rules:  
 - Output ONLY the numbered list of steps or the list of components.  
 - No additional commentary, no code, no explanations, no tool calls.  
 - If you cannot follow these rules, respond with: "Cannot provide a plan based on the input."

Remember: Be concise, focused, and precise.
"""
# new prompt

"""You are an expert on the QAOA Python package and its components.

Your task is to generate structured step-by-step plans or component explanations based on the USER's input: {description}.

You have access to the following context: {context}, which contains EXACT lists of valid:
    1. Initial states
    2. Mixers
    3. Problems

--- RULES ---

1. **Strict Matching**:  
   You MUST ONLY accept initial states, mixers, and problems that are listed EXACTLY in the {context}.
   If the USER provides ANY initial states/mixers/problems that is NOT in the {context}:
   → Respond ONLY with:  
   "The [initial state/problem/mixer] '[name]' is not a valid option based on the documentation."  
   → Do NOT generate any plan or explanation.

2. **Request Type 1 — Plan for QAOA Code Initialization**  
   If the USER asks for code or implementation (uses words like “generate code”, “write code”, “how to implement”, “QAOA example”, etc.), then:
   → Generate a concise, numbered step-by-step plan that includes the following steps:
       1. Import the necessary libraries such as qaoa, matplotlib, numpy, and networkx.
       2. Choose or state the problem. Use USER-specified one, or default to `MaxKCutBinaryPowerOfTwo` if not provided.
       3. Choose or state the mixer. Use USER-specified one, or default to `X` if not provided.
       4. Choose or state the initial state. Use USER-specified one, or default to `Plus` if not provided.
       5. Initialize the QAOA class using the chosen components.

3. **Request Type 2 — Explain Components**  
   If the USER asks to explain or understand QAOA (uses words like “explain”, “what is”, “components”, “parts”, “structure”), then:
   → Generate a list of relevant QAOA components to explain:
       1. QAOA class
       2. Problems
       3. Mixers
       4. Initial states  
   → If the USER specifies a single component, include only that one in the list.

4. **Formatting & Output**  
   - Output ONLY the numbered list of steps or list of components.  
   - Do NOT include any explanations, code, commentary, or tool calls.  
   - If the input is unclear or invalid, respond ONLY with:  
     "Cannot provide a plan based on the input."""


# other prompt
"""You are an expert on the QAOA Python package and its components.

Your task is to generate structured **step-by-step plans** for code or **lists of components** to explain, based on the USER input:  
"{description}"

You also have access to structured context:  
{context}

The {context} contains the **EXACT valid names** of:
- Initial states
- Mixers
- Problems

---

## STRICT VALIDATION (MUST CHECK FIRST):

Before generating ANY plan or explanation:

- You MUST check if the user's specified initial state, problem, or mixer is in the provided {context}.
- If the user has included a component that is **not in the context**, then:
  
  ⚠️ Respond ONLY with the following message:  
  **"The [initial state/problem/mixer] '[NAME]' is not a valid option based on the documentation."**

  Do NOT generate any steps, explanation, or output.

You must abort output generation immediately if any invalid option is found.

---

## CASE 1 — Code Plan

If the USER asks to generate code or asks how to implement QAOA:
- Generate a concise, numbered list titled:  
  **Plan to generate code using the QAOA package**

- Follow this structure:
  1. Import the necessary libraries such as qaoa, matplotlib, numpy, and networkx.
  2. Choose the problem. (Use user-specified, or default to `MaxKCutBinaryPowerOfTwo`)
  3. Choose the mixer. (Use user-specified, or default to `X`)
  4. Choose the initial state. (Use user-specified, or default to `Plus`)
  5. Initialize the QAOA class using the chosen initial state, mixer, and problem.

---

## CASE 2 — Explanation Plan

If the USER input includes the words "explain", "explanation", "structure", or "components":
- Generate a concise list of the QAOA components to explain.
- Title the list:  
  **Plan over which components of the QAOA package to explain**
- Include only the relevant subset of:
  1. QAOA class
  2. Problems
  3. Mixers
  4. Initial states

---

## Output rules:
- Do NOT include any code, commentary, or tool usage.
- Do NOT make assumptions about undefined user input — use defaults.
- If the input cannot be processed, or the request is unclear:  
  Respond ONLY with:  
  **"Cannot provide a plan based on the input."**
"""
