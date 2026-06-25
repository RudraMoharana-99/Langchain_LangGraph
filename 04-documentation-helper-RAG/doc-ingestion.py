import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from .logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model= "text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10
)
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLS into chunks of specified size."""
    chunks = []

    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)

    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            Colors.BLUE
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents"
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"Tavily Extract: Start concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN
    )
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0

    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception {result} ")
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page['raw_content'],
                    metadata = {"source": extracted_page['url']}
                )
                all_pages.append(document)

    return all_pages


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN
    )

    #====================================
    #========Create batches==============
    #====================================
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f" VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    #==================================
    # Process all batches concurrently
    #==================================
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await asyncio.to_thread(
                            vectorstore.add_documents,
                            batch
                        )
                
            log_success(
                f"Vectorstore Indexing: Successfully addes batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"Vectorstore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    #================================
    #==Process batches concurrently==
    #================================
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #================================
    #====Count seccessful batches====
    #================================
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed succeefully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed ({successful}/{len(batches)}) batches successfully"
        )

async def main():
    """Main async fuction to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://docs.langchain.com/",
        Colors.PURPLE
    )

    # Crawl the documentation site

    # res = tavily_crawl.invoke(
    #     {
    #         "url": "https://docs.langchain.com/",
    #         "max_depth": 1,
    #         "extract_depth": "advanced",
    #         # "instructions": "content on AI agent",
    #     }
    # )
    # all_docs = [Document(page_content=result['raw_content'], metadata={"source": result["url"]}) for result in res['results']]

    site_map = tavily_map.invoke("https://docs.langchain.com/")

    log_success(
        f"TavilyCrawl: Successfully crawled {len(site_map['results'])} URLS from documentatio site"
    )


    #==============================================================
    #============Split urls to batches of 20=======================
    #==============================================================
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    log_info(
        f" URL Processing: Split {len(site_map['results'])} URLS into {len(url_batches)} batches",
        Colors.BLUE,
    )

    #==============================================================
    #============Extract documents from URLs=======================
    #==============================================================
    all_docs = await async_extract(url_batches=url_batches)
    print(type(all_docs))


    #==============================================================
    #==============Split documents into chunks=====================
    #==============================================================

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents with 3000 chunk size with 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    #==============================================================
    #==============Process documents asynchronously================
    #==============================================================
    await index_documents_async(splitted_docs, batch_size=100)

    log_header("PIPELINE COMPLETE")
    log_success(" Documentation ingestion pipeline finished successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f"URLs mapped: {len(site_map['results'])}")
    log_info(f"Documents extracted: {len(all_docs)}")
    log_info(f"Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())