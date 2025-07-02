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
You are an expert in the QAOA Python package and a code assistant. Your goal is to make a prompt that is a plan (either for code generation or for generating text with information about the QAOA) to another agent. 
You will be concise and focused on implementation steps for the agent. You will not respond with anything that is not a specific step.

Given this task: "{description}", you will do only 1 of the 2 following things (and only 1 of them):
    1. If the user asks for a specific code or how to implement the package, you will respond with the plan for how to generate the code by create a numbered list of steps to generate a QAOA Python script using only the QAOA package as far as possible.
    2. If the user asks for an explanation of the QAOA package, you will provide a brief list over which components and parts of the QAOA package the other agent will generate an text for.

If you are unsure of which of the two possiblities is the correct to do, you can assume it is 1. 

Instructions for 1.
 - Each step should be implementation-focused and concise. Do NOT call tools or generate code — just describe how to do it.
 - If the user states the initial state, problem, or mixer, you can respond with the plan for how to generate the code using those components.
 - If not, you will by default assumme that the user meant the example combination of problem MaxKCutBinaryPowerOfTwo, initial state Plus, and mixer X.

Instructions for 2.
 - Provide a list over which components and parts of the QAOA package the other agent will generate an text for.
 - You will not generate the explanations yourself, but you will create a list of components (for example important classes and methods) that the other agent will generate text for.
""",
)

planner_chain = LLMChain(llm=llm, prompt=planner_prompt)

# -----Making the Code Agent-----
code_agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# ----- Main loop -----
while True:
    query = input("Ask a question (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    plan = planner_chain.run({"description": query})
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
