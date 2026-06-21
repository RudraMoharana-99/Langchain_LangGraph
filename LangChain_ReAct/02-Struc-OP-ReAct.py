from dotenv import load_dotenv

load_dotenv()
from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the Agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's annswer to the query.")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 remote active job posting for an ai engineer using LangChain in Any location on linkedin list their details")})
    print(result)
    
if __name__ == ("__main__"):
    main()