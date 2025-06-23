import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from connect_memory_with_llm import load_llm, load_vector_store, set_custom_prompt
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(
    page_title="Medical Assistant Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #475063;
        color: white;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
    }
    .stTextInput>div>div>input {
        background-color: #2b313e;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🏥 Medical Assistant Chatbot")
st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Your AI-powered medical assistant. Ask any medical questions and get accurate, 
        evidence-based answers from our medical knowledge base.
    </div>
    """, unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("""
        <div style='text-align: center;'>
            <h2>ℹ️ About</h2>
            <p>This chatbot uses advanced AI to provide medical information based on reliable sources.</p>
            <hr>
            <h3>📚 Knowledge Base</h3>
            <p>Powered by medical encyclopedia and verified medical resources.</p>
            <hr>
            <h3>⚠️ Disclaimer</h3>
            <p>This is not a substitute for professional medical advice. Always consult healthcare providers for medical decisions.</p>
        </div>
    """, unsafe_allow_html=True)

# Initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    try:
        # Initialize LLM
        llm = load_llm()
        if not llm:
            st.error("Failed to initialize the language model. Please check your HuggingFace token.")
            return None

        # Load vector store
        db = load_vector_store()
        if not db:
            st.error("Failed to load the knowledge base. Please run create_memory_for_llm.py first.")
            return None

        # Create prompt
        prompt = set_custom_prompt()

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

# Initialize the chatbot
qa_chain = initialize_chatbot()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your medical question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # If this is the first message and it's a greeting, give a warm welcome
    greetings = ["hi", "hello", "hey"]
    if len(st.session_state.messages) == 1 and prompt.strip().lower() in greetings:
        welcome_message = "👋 Hello! Welcome to the Medical Assistant Chatbot. How can I help you today? Feel free to ask any medical question."
        with st.chat_message("assistant"):
            st.markdown(welcome_message)
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    else:
        # Build context from previous messages
        history = ""
        for msg in st.session_state.messages[-6:]:  # Use last 6 messages (3 Q&A pairs)
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        history += f"User: {prompt}\n"
        
        # Get bot response
        if qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({"query": prompt, "context": history})
                    st.markdown(response["result"])
                    
                    # Add bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                    
                    # Display source documents in an expander
                    with st.expander("View Sources"):
                        for doc in response["source_documents"]:
                            st.markdown(f"**Source:** {doc.page_content[:200]}...")
        else:
            st.error("Chatbot is not properly initialized. Please check the error messages above.")

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem; padding: 1rem; border-top: 1px solid #ddd;'>
        <p>© 2024 Medical Assistant Chatbot | Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def set_custom_prompt():
    custom_prompt_template = """
    You are a medical assistant. Use the following conversation history and retrieved context to answer the question.
    If the answer is not in the context, politely say 'I don't know' and do not make up information. Use three sentences maximum and keep the answer concise.
    Do not provide anything outside of the context.

    Conversation history:
    {context}
    Current question: {question}

    Answer:"""
    
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt
