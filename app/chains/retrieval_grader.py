from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = "gpt-4o", temperature=0)

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# LLM with function call 
structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)

# Prompt 
system = """You are an expert grader assessing relevance of a given answer to a user question. \n 
    If the provided answer answers the users query, grade it as relevant and valid. \n
    Give a binary score 'yes' or 'no' score to indicate whether the answer is resolving the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", " User question: {question} \n\n Provided answer: \n\n {answer}")
    ]
)

answer_grader = grade_prompt | structured_llm_grader_docs
