# LangGraph ChatBot

## Overview
LangGraph ChatBot is a Streamlit-based web application that integrates LLMs to interact with users dynamically. It leverages the llama_index library for document retrieval and query handling, enhancing user interaction by providing intelligent responses based on a rich document set. We use LlamParser for unstructured pdf processing, LangGraph and LLama Index to make use of Agentic RAG workflow

## Features
- Upload PDF documents for context-aware conversation.
- Query large language models for information extraction.



## Environment Setup
Create an `.env` file in the root of the project directory with the necessary environment variables:


OPENAI_API_KEY=sk-proj-yourkey \n
LLAMA_PARSE=llx-yourkey \n
LANGCHAIN_API_KEY=lsv2_yourkey (if want to track the trace) \n



## Installation and run locally

Clone the repository:

git clone https://your-repository-url.git

cd LangGraph-ChatBot
Install the required Python packages:


## Dockerization 

run Docker command 

The application is dockerized for easy deployment and isolation.

Building the Docker Image
Build the Docker image using the following command:

`docker build -t langgraph-chatbot .`

## Running the Application
To run the application, execute:


`ddocker run -it -p 8083:8083 --env-file .env --name langgraph-chatbot langgraph-chatbot`




This command will start the Streamlit server and open the application in your default web browser.

## Usage
Upload Documents: Start by uploading the PDF documents you want the chatbot to reference.
Ask Questions: Enter your questions in the chat input area. The chatbot uses the uploaded documents to generate context-aware responses.

