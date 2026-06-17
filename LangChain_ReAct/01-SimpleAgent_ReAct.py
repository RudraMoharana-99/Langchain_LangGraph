from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that search over the internet
    Args:
        query: the query to search for
    
    results:
        The search results
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)



llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 remote active job posting for an ai engineer using LangChain in Any location on linkedin list their details")})
    print(result)
    
if __name__ == ("__main__"):
    main()