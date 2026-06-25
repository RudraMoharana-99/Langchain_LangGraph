import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


load_dotenv()

#===============================================================
#========Initialize embeddings (same as ingestion.py)===========
#===============================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#===============================================================
#==================Initialize VectorStore=======================
#===============================================================
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings,
    )

#===============================================================
#==================Initialize ChatModel=========================
#===============================================================
model = init_chat_model(model="gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrive_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retriever= vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_doc = retriever.invoke(query)
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_doc
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_doc


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question
    
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    system_prompt = (
        """You are a concise AI assistant that answers questions about LangChain documentation using retrieved sources.

            Your role:
            - Answer only questions within LangChain scope (components, chains, agents, integrations, API reference)
            - Decline questions outside LangChain or about competing frameworks
            - Use the retrieval tool to find relevant documentation before answering every question

            When using the retrieval tool:
            - Search for specific components, methods, or concepts mentioned in the question
            - If multiple results exist, prioritize official API docs over tutorials
            - If results conflict or seem outdated, acknowledge the ambiguity

            Citation format:
            - Include the source document title and section in parentheses after the claim
            - Example: "LangChain uses LCEL syntax (LangChain: LangChain Expression Language guide)"

            If you cannot find an answer:
            - State clearly: "I could not find this in the LangChain documentation."
            - Suggest what the user might search for instead
            - Do not guess or infer beyond what the docs contain
    """
    )

    agent = create_agent(model=model, tools=[retrive_context], system_prompt=system_prompt)

    # Build messages list
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]
    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result)