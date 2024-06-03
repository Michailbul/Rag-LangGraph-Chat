# LangGraph ChatBot

## Overview
LangGraph ChatBot is a Streamlit-based web application that integrates language models to interact with users dynamically. It leverages the llama_index library for document retrieval and query handling, enhancing user interaction by providing intelligent responses based on a rich document set.

## Features
- Upload PDF documents for context-aware conversation.
- Query large language models for information extraction.
- Real-time response generation with AI-enhanced capabilities.

## Prerequisites
Before you start, ensure you have the following installed:
- Python 3.8 or newer
- pip (Python package installer)

## Installation

Clone the repository:
```bash
git clone https://your-repository-url.git
cd LangGraph-ChatBot
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Environment Setup
Create a .env file in the root directory of your project and populate it with your keys:

plaintext
Copy code
QDRANT_URL=<your-qdrant-url>
QDRANT_API_KEY=<your-qdrant-api-key>
LLAMA_PARSE=<your-llama-parse-key>
OPENAI_API_KEY=<your-openai-api-key>
LANGCHAIN_API_KEY=<your-langchain-api-key>
BUCKET_NAME=<your-s3-bucket-name>
Running the Application
To run the application, execute:

bash
Copy code
streamlit run main.py
This command will start the Streamlit server and open the application in your default web browser.

Usage
Upload Documents: Start by uploading the PDF documents you want the chatbot to reference.
Ask Questions: Enter your questions in the chat input area. The chatbot uses the uploaded documents to generate context-aware responses.
Contributing
Contributions to the LangGraph ChatBot are welcome. Please ensure to follow the existing coding style, update tests as appropriate, and update the README with relevant details.

License
MIT


### Additional Recommendations
- **Screenshots**: Adding screenshots of your application in action can greatly improve the visual appeal of your README.
- **Demo Video**: If possible, include a link to a video demonstrating the usage of your chatbot.
- **Code Snippets**: Include examples of how to use the chatbot, especially if it can be integrated or interacted with programmatically.
- **FAQs or Troub
