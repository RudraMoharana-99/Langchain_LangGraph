import os


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from operator import itemgetter

load_dotenv()

print(" [Initializing components] ....")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based ONLY on the following context.
    
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into single strings."""
    return "\n\n".join(doc.page_content for doc in docs)


#=============================================================
# Option 1: Use implementation WITHOUT LCEL
#=============================================================
def retrieval_chain_without_lcel(query: str):
    """Simple retrival chain without LCEL
    
    Limitations:
    - Manual step by step execution
    - No built-in streaming support
    - No aync support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format document into context string
    context = format_docs(docs=docs)

    # Step 3: Format the prompt with context and query
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messeges
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

#=======================================================================
# Option 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
#=======================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LANGCHAIN Expression Language).
    Returns a chin that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less coe: More concise and readable
    - Reusable: Chain can be saved, shared and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_cahin = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        |StrOutputParser()
    )
    return retrieval_cahin

if __name__ == "__main__":
    print( "    [Retrieving] ....")

    # Query
    query = "What is pinecone in machine learning?"

    #=============================================================
    # Option 0: Raw incocation without RAG
    #=============================================================
    print("\n" + "="*70)
    print(" [IMPLEMENTATION 0]: Raw invocation (No RAG)")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\n[ANSWER]: ")
    print(result_raw.content)

    #=============================================================
    # Option 1: Use implementation WITHOUT LCEL
    #=============================================================
    print("\n" + "="*70)
    print(" [IMPLEMENTATION 1]: Without (LCEL)")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query=query)
    print("\n[ANSWER]: ")
    print(result_without_lcel)

    #=======================================================================
    # Option 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
    #=======================================================================
    print("\n" + "="*70)
    print(" [IMPLEMENTATION 2]: With (LCEL) Better approach")
    print("="*70)
    print("Why LCEL is Better:")
    print("- More concise and readable")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke() and chain.astream()")
    print("="*70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\n[ANSWER]: ")
    print(result_with_lcel)