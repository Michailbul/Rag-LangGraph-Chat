import os
from dotenv import load_dotenv
import boto3
load_dotenv()  # This will load all the variables from a .env file into the environment

def setup_environment():

    
    os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL")
    os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
    os.environ["LLAMA_PARSE"] = os.getenv("LLAMA_PARSE")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "TEMUS TEST"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['UNSTRUCTURED_API_KEY'] = os.getenv("UNSTRUCTURED_API_KEY")



def get_s3_client():
    return boto3.client('s3')

def get_bucket_name():
    return os.getenv("BUCKET_NAME")