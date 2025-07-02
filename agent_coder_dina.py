# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain and OpenAI imports
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
import re

# Instantiate the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

@tool
def execute_code(code: str) -> str:
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


# List of tools for the agent
tools = [execute_code]

# Define prompt
code_prompt = PromptTemplate(
    input_variables=["input"],
    template="""You are a Python code testing assistant.

You're given Python code. Your task is to **run** the code and report **any errors** that occur during execution. Use the `execute_code` tool to help with this.

Code:
{input}
"""
)

# Sample code to test
code_to_test = """
import networkx as nx
import numpy as np
from qaoa.problems import MaxKCutBinaryPowerOfTwo

G = nx.Graph()
G.add_nodes_from(np.arange(0, 5, 1))  # Create a graph with 5 nodes
G.add_weighted_edges_from(
    [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (3, 4, 1.0), (4, 2, 1.0)]
)

problem = MaxKCutBinaryPowerOfTwo(G=G, k_cuts=2)
result = problem  # Store result to be returned
"""

# Initialize the agent
code_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent with the code to test
answer = code_agent.run(f"Please run the following code and report any errors:\n\n{code_to_test}")

# Output the result
print(answer)
