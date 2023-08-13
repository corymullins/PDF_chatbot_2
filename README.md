# Chatbot for Multiple PDFs

![GitHub repo size](https://img.shields.io/github/repo-size/corymullins/PDF_chatbot_2)
![GitHub last commit](https://img.shields.io/github/last-commit/corymullins/PDF_chatbot_2)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/corymullins/PDF_chatbot_2/main/app.py)

A Streamlit-based chatbot that allows users to ask questions and retrieve information from multiple PDF documents using NLP techniques and vector indexing.

## Features

- Extracts text content from uploaded PDF files.
- Splits text into overlapping chunks for efficient processing.
- Creates a searchable vector index using Qdrant.
- Implements a conversational chain for retrieving answers.
- Provides a user-friendly interface using Streamlit.

## Getting Started

### Prerequisites

- Python 3.x
- [Streamlit](https://streamlit.io)
- [Qdrant Cloud API key](https://cloud.qdrant.io/)
- [OpenAI API key](https://beta.openai.com/account/api-keys)
- [Other libraries listed in requirements.txt]

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/corymullins/PDF_chatbot_2.git
   cd PDF_chatbot_2

2. Install dependencies:
   ```sh
   pip install -r requirements.txt

3. Copy `.env.example` to `.env` in the root directory, and add your environment variables:
   ```makefile
   QDRANT_HOST=<your-qdrant-host>
   QDRANT_API_KEY=<your-qdrant-api-key>
   QDRANT_COLLECTION_NAME=<your-qdrant-collection-name>

### Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py

2. Visit the URL provided in your terminal to access the chatbot interface.

3. In the sidebar, upload PDFs and click "Process" to initialize the chatbot.

## Code overview

**app.py**
- get_pdf_text - Extract text from PDFs
- get_text_chunks - Split text into chunks
- get_vectorstore - Create Qdrant index
- get_conversation_chain - Build conversational chain
- handle_userinput - Process user questions
- main - Streamlit app and UI

**htmlTemplates.py**
- HTML/CSS templates for chatbot UI

## License

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the user interface.
- [Qdrant](https://qdrant.github.io/qdrant/) for the vector store.
- [OpenAI GPT-3](https://openai.com/gpt-3/) for natural language processing.
