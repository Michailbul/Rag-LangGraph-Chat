from typing import List, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from typing import List, TypedDict, Optional, Union



class SessionState(TypedDict):
    """
    Represents the session state which includes the agent, file paths, and chat history.
    """
    agent: Optional[str]  # Assume this is the type for your agent
    file_paths: List[str]
    chat_history: List[Union[HumanMessage, AIMessage]]
    



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Gonna be modified in every node of out graph
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    session_state: SessionState

    question: str
    generation: str
    #web_search: bool
    documents: List[str]
    chat_history: List[Union[HumanMessage, AIMessage]]