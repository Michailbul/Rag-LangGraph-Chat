from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

### Answer enhancer

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Prompt
system = """You an answer re-writer that  converts a given input answer for a user's question to a better, enhanced version that is providing \
    sufficient rich answer. You should use the provided context for more background information. Your goal is to improve the answer quality and make it more informative.
    If the question is not neeed improvement and is already perfect, just repeat it as it is.
    """

    #TODO 
    # We hardcoded the decisionmaking in prompt --> use validation via pydantic and conditional edges instead

enhancer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the input question: \n\n {question} \n .Context: {context} \n Provide your refined answer:",
        ),
    ]
)

enhancer_chain = enhancer_prompt | llm | StrOutputParser()

