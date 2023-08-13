import streamlit as st
import qdrant_client
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import OpenAI

def get_pdf_text(pdf_docs):
    """
    Extracts and combines text from a list of PDF documents.

    Args:
        pdf_docs (list): List of file paths of PDF documents
    
    Returns:
        text (str): String containing extracted text from all PDFs
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {pdf}\n{e}")
    return text


def get_text_chunks(text):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): Raw text to split
    
    Returns:
        text_chunks (list): List of chunked strings
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_vectorstore(text_chunks):
    """
    Creates a vectorstore search index from text chunks.

    Args:
        text_chunks (list): List of text chunks
    
    Returns:
        Vectorstore: Searchable vectorstore index
    """
    try:
        client = qdrant_client.QdrantClient(
            os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    
        embeddings = OpenAIEmbeddings()

        vectorstore = Qdrant(
            client=client, 
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
            embeddings=embeddings,
        )
    
        vectorstore.add_texts(text_chunks)
    except qdrant_client.QdrantClientException as e:
        st.error(f"Error connecting to Qdrant: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Creates a conversational chain using the vectorstore.

    Args:
        vectorstore (Vectorstore): Vectorstore index
    
    Returns:
        ConversationChain: Conversational chain 
    """
    llm = OpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user question and displays chat history.

    Args:
        user_question (str): User's question
    """
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def main():
    """
    Main Streamlit app entry point. Sets up Streamlit interface and handles user interactions.
    """
    load_dotenv()
    
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()