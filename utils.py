from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_parse import LlamaParse

from pprint import pprint
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
import textwrap


import os 
import logging

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

def create_nodes(file_path):

    instruction = """The provided document is a financial report of a large company.
    This form provides detailed financial information about the company's performance.
    It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures.
    It contains many tables.
    Be precise while answering the questions.
    Always provide reasoning over your answer, providing sources and all the relevant information that led to this answer"""

    parser = LlamaParse(
            api_key=os.environ["LLAMA_PARSE"],
            result_type="markdown",
            parsing_instruction=instruction,
            max_timeout=5000,
        )

    file_extractor = {".pdf": parser}


    logging.warning(f"__________________________ PROCESSING DOCUMENTS ______________________")


    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor ).load_data()

    logging.info(f"documents {documents}")


    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    
    return nodes



def create_vector_index(file_path):

    nodes = create_nodes(file_path)

    vector_index = VectorStoreIndex(nodes)

    return vector_index, nodes



def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    vector_index,nodes = create_vector_index(file_path)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the document.
    
        Useful if you have specific questions over the provided document.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            ),
            response_mode="refine"
        )
        response = query_engine.query(query)
        return response
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of document. "
            "Do NOT use if you have specific questions over document."
        ),
    )

    return vector_query_tool, summary_tool



def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

    
def print_reranked_docs(reranked_docs):
    for doc in reranked_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {doc.metadata['relevance_score']}")
        print("-" * 80)
        print()

def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))