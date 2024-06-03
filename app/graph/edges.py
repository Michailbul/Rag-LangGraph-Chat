from typing import Any, Dict

from app.graph.state import GraphState
from langchain.schema import Document
from app.chains.router import question_router
from app.chains.hallucination_grader import hallucination_grader
from app.chains.answer_grader import answer_grader
from pprint import pprint

### Edges

def route_question(state):
    """
    Route question to answer or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(f"question is {question}")
    source = question_router.invoke({"question": question})   
    if source.datasource == 'generate_casual':
        print("DECISION:---ROUTE TO STRAIGHT ANSWER---")
        return "generate_casual"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, HALLUCINATE RE-TRY---")
        return "hallucinate"






#TODO adapt this to the new system
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


#NOTE workarounded/replaced with hallucination_grader
def grade_answer(state):
    """
    Determines whether the answer is relevant to the question and resolves it .

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK ANSWER RELEVANCE TO QUESTION ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = answer_grader.invoke(
        {"question": question, "answer": generation}
    )
    grade = score.binary_score
    if grade == "yes":
        print("---GRADE: ANSWER RELEVANT---")
        return "useful"
    else:
        print("---GRADE: ASWER NOT RELEVANT---")
    return "not useful"
        
