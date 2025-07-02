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

from qaoa.initialstates.dicke1_2_initialstate import Dicke1_2


# Tools for the Planner Agent
@tool
def setup_for_example(description: str) -> str:
    """
    Sets up a plan for an example of a QAOA instance/problem based on the description string.
    """
    return f"Setup for example with description: {description}"


# Tools for the Code Agent
@tool
def make_initialstate(description: str) -> str:
    """
    Creates an initial state object based on the description string.
    Availible options: Dicke, Dicke1_2, LessthanK, maxkcut, plus, statevector
    """
    if "dicke" in description.lower():
        state = Dicke1_2()
        return f"Created Dicke1_2 instance: {state}"  # return a string representation


@tool
def make_problem(description: str) -> str:
    """
    Creates a problem object based on the description string.
    Availible options: MaxCut, QUBO, ExactCover, Portfolio
    """
    # Placeholder for actual problem creation logic
    return f"Problem created with description: {description}"


@tool
def generate_code(description: str) -> str:
    """
    Generates code based on the description string.
    """
    # Placeholder for code generation logic


# Use the LLM from OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

code_tools = [make_initialstate, make_problem]
planner_tools = []

# initializes the agent with the tools, LLM, and agent type
code_agent = initialize_agent(
    code_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# planner_prompt = PromptTemplate(input_variables=["query"], template="You are a planner agent and an expert in the QAOA python package. Create a step by step plan for how to generate a code using the QAOA package and that is based on the following description: {query}")
# planner_agent = create_react_agent(llm=llm, tools=planner_tools, prompt=planner_prompt)
# planner_agent_executor = AgentExecutor(agent=planner_agent, tools=planner_tools, verbose=True)
prompt = PromptTemplate(
    input_variables=["description"],
    template="You are a quantum software expert. You are an expert in the QAOA python package. You will only use the QAOA package to solve the problem as far as that is possible. "
    "Given this task description: {description}, generate a numbered list of steps to generate a code using the QAOA package. "
    "Each step should be one sentence. Keep it implementation-focused.",
)

planner_agent = initialize_agent(
    code_tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    prompt=prompt,
)

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
