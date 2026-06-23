import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print(" [Ingesting] ....")

    loader = TextLoader("03-RAG-GIST/mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print(" [Splitting] ....")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=800,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents=document)
    print(f"    [Created] {len(texts)} chunks")

    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print(" [Ingesting] ....")
    PineconeVectorStore.from_documents(documents=texts, embedding=embedding, index_name=os.environ['INDEX_NAME'])
    print(" [Finsh] ....")
