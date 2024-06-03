from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model = "gpt-4-turbo", temperature=0,)


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'.")
 
# LLM with function call 
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
 
# Prompt 
system = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. \n 
     Restrict yourself to give a binary score, either 'yes' or 'no'. If the answer is supported or partially supported by the set of facts, consider it a yes. \n
    Don't consider calling external APIs for additional information as consistent with the facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
]
)
  
hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination
