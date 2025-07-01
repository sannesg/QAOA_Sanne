# Loading API key
from dotenv import load_dotenv

load_dotenv()

# To use the OpenAI LLM
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI

from qaoa.initialstates.dicke1_2_initialstate import Dicke1_2


# Tools for the agent
@tool
def make_initialstate_Dicke1_2(tool_input: str) -> str:
    """Creates an initial state Dicke1_2"""
    dicke_state = Dicke1_2()  # create the instance
    return f"Created Dicke1_2 instance: {dicke_state}"  # return a string representation


# Use the LLM from OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [make_initialstate_Dicke1_2]

# initializes the agent with the tools, LLM, and agent type
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

query = "Make an initial state Dicke1_2"
response = agent.run(query)
print(response)
