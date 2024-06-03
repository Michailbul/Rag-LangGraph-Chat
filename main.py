import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from langchain_core.messages import HumanMessage, SystemMessage
import os
import logging
from app.graph.graph import app
from pathlib import Path
import cred
import os
from dotenv import load_dotenv
from utils import get_doc_tools


load_dotenv()



def main():

    add_custom_css()

    cred.setup_environment()

    # s3_client = cred.get_s3_client()
    # bucket_name = cred.get_bucket_name()

    st.title("Multi PDF Chat with financial data")


    # Initialize session state for chat history and file paths
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am your assistant. How can I help you?")]
    if "file_paths" not in st.session_state:
        st.session_state.file_paths = []  # Initialize file_paths in session state
    if "agent" not in st.session_state:
        st.session_state.agent = None


    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)
        process = st.button("Process")


    if process:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            setup_agent(uploaded_files)

    # Handle user input and display conversation using chat_message
    user_query = st.chat_input("Type your message here...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        if st.session_state.agent:


            response = handle_user_input(user_query)
            answer = response['generation']

            #TODO add sources to the response
            #source = return_sources(response)

            st.session_state.chat_history.append(AIMessage(content = str(answer)))

            
                

    # Display the conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
            
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)


    
def add_custom_css():
    custom_css = """
    <style>
    .stMarkdown p {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        word-wrap: break-word;
        white-space: normal;
    }
    .stChatMessage {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        word-wrap: break-word;
        white-space: normal;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def handle_user_input(user_question):
    # Query the pre-initialized agent for a response
    if "agent" in st.session_state and st.session_state.agent is not None:

        session_state = {
        'agent': st.session_state.agent,
        'file_paths': [],  
        'chat_history': st.session_state.chat_history  
        }
       
        
        result = app.invoke({"question": user_question, "session_state" : session_state})
      
    return result




def setup_agent(uploaded_files):

    llm = OpenAI(model="gpt-4-turbo", temperature=0)

    temp_dir = './temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    file_paths = [save_file(uploaded_file, temp_dir) for uploaded_file in uploaded_files]
    file_to_tools_dict = {file_path: get_doc_tools(file_path, Path(file_path).stem) for file_path in file_paths}
   
    initial_tools = [tool for tools in file_to_tools_dict.values() for tool in tools]
    agent_worker = FunctionCallingAgentWorker.from_tools(initial_tools, llm=llm, verbose=True)

    st.session_state.agent = AgentRunner(agent_worker)
    st.write("Agent is set up and ready to answer questions.")



def save_file(uploaded_file, temp_dir):
    file_path = os.path.join(temp_dir, uploaded_file.name)
    # Check if the file already exists and delete it if it does
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(f"Deleted existing file: {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    logging.info(f"Processed uploaded file: {uploaded_file.name}")
    return file_path

    

def get_parser():
    instruction = """
    The provided document is a financial report of a large company.
    This form provides detailed financial information about the company's performance.
    It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures.
    It contains many tables.
    Be precise while answering the questions."""
    return LlamaParse(
        api_key=os.getenv("LLAMA_PARSE"),
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
    )




def return_sources(response):
    sources = []
    logging.info(f"__________________ENTERING  THE RETURN SOURCE FUNC __________________-")
    logging.info(f"RESPONSE ENTERING THE RETURN SOURCE FUNC: {response}")
    # Iterate over each NodeWithScore in the response
    if response.source_nodes is not None:
        for node_with_score in response.source_nodes:
            logging.info(f"#####   response.source_nodes  ##### : {response.source_nodes}")
            node = node_with_score.node
            # Extract metadata from the node if available
            if hasattr(node, 'metadata'):
                logging.info(f"NODE METADATA: {node.metadata}")
                metadata = node.metadata
                file_info = {
                    "File Name": metadata.get("file_name"),
                    "Page label": metadata.get("page_label"),
                    "Score": node_with_score.score,

                }
                sources.append(file_info)
                print(f"Source Document: {file_info}")


        return None
    
    return sources


if __name__ == "__main__":
    main()        

   
        