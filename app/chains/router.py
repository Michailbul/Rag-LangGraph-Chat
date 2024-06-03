### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os
load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

### Router

# Data model
class RouteQuery(BaseModel):
    """Route a user query."""

    datasource: Literal["vectorstore", "generate_casual"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or generate casual answer directly if it is not related to documents and finance",
    )

# LLM with function call 
llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt 
system = """You are an expert at routing a user question to a vectorstore or generate_casual answer without retrieval.
The vectorstore contains documents related to Financial data of Large companies.
Use the vectorstore for questions on these topics and related topics. For all else, provide an answer in a casual manner"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


