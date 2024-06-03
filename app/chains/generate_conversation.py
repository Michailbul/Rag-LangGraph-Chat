from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0)

### Generate

from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = ChatPromptTemplate.from_template(
    """You are  helpfull and friendly assistant. Please provide conversational answer to the queries 
Query: {question}
Answer:"""
)
 
# Chain
conversational_chain = prompt | llm | StrOutputParser()
