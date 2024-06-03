from typing import List
from app.chains.generate_rag import rag_chain
from app.chains.generate_conversation import conversational_chain
from app.chains.answer_grader import answer_grader
from app.chains.enhancer import enhancer_chain
from langchain.schema import Document
from app.chains.question_rewriter import question_rewriter
import streamlit as st



def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
   
    
    question = state.get("question", "")
    session_state = state.get("session_state", {})

    # Access the agent and chat_history from session_state
    agent = session_state.get("agent", None)

    response = agent.query(question)
  
    documents = nodes_to_documents(response)

    return {"documents" : documents , "question": question, "session_state" : session_state, "generation": response}  



def nodes_to_documents(response):
    """
    Convert response source nodes to documents

    Args:
        response (Response): The response object from the agent

    Returns:
        documents (List[Document]): List of documents
    """
    keys_to_keep = {'page_label', 'file_name'}

    # Create documents list from response source nodes
    documents: List[Document] = [
        Document(
            page_content=node.text,
            metadata={key: node.metadata[key] for key in keys_to_keep if key in node.metadata},
            score=node.score
        )
        for node in response.source_nodes
    ]
    return documents



def response_enhancer(state):
    """
    Improves the quality of LLamaIndex Agent system

    Reads the provided documents and peraphrases the answer leveraging the Retrieved data
    
    
    """ 
    print("---ENHANCING THE ANSWER---")

    question = state["question"]
    documents = state["documents"]


    generation = enhancer_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}



def generate_rag(state):
    """
    # Here we are not using LLamaIndex Agent system but swithcing to RAG for answer generation using our own simple chain


    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE RAG Answer---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}




def generate_casual(state):
    """
    Generate answer withouth RAG over financial documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE CASUAL Answer---")
    question = state["question"]
    #documents = state["documents"]
    
    # RAG generation
    generation = conversational_chain.invoke({ "question": question})
    return {"question": question, "generation": generation}



def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    original_answer = state['answer']

    # Re-write question
    better_question = question_rewriter.invoke({"question": question, "context": documents, "answer": original_answer})
    return {"documents": documents, "question": better_question}