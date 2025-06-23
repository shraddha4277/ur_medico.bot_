from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os
import traceback

def load_llm():
    try:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Please set your HuggingFace token as an environment variable named 'HF_TOKEN'")

        # Using a different model that might be more accessible
        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        print("Initializing LLM with Mistral-7B-Instruct-v0.3 model...")
        
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=HF_TOKEN,
            task="text2text-generation",
            temperature=0.5,
            max_new_tokens=512,
            top_p=0.95,
            repetition_penalty=1.15
        )
        print("LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return None

def set_custom_prompt():
    custom_prompt_template = """
You are acting as a medical information assistant, not a doctor.
Your guidelines:
Be precise: Give fact-based, evidence-supported information only.
Never hallucinate or make up facts — if you’re unsure, politely say so.
Never prescribe medicines, dosages, or treatments under any circumstance.
Never suggest self-medication.
If a user asks for a prescription, diagnosis, or drug recommendation, politely decline and remind them to consult a qualified doctor.
Always maintain a polite, supportive, and responsible tone.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Don't provide anything out of the context.

    Context: {context}
    Question: {question}

    Answer:"""
    
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_vector_store():
    try:
        print("Loading FAISS database...")
        DB_FAISS_PATH = "vectorstore/db_faiss"
        
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}. Please run create_memory_for_llm.py first.")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Database loaded successfully")
        return db
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None

def main():
    try:
        # Initialize LLM
        llm = load_llm()
        if not llm:
            return

        # Load vector store
        db = load_vector_store()
        if not db:
            return

        # Create prompt
        prompt = set_custom_prompt()

        # Create QA chain
        print("Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        print("QA chain created successfully")

        print("\nMedical Chatbot is ready! Type 'quit' to exit.")
        while True:
            # Get user query
            USER_QUERY = input("\nEnter your medical question: ").strip()
            
            if USER_QUERY.lower() == 'quit':
                print("Goodbye!")
                break
                
            if not USER_QUERY:
                print("Please enter a question.")
                continue
            
            # Get response
            print("\nGenerating response...")
            response = qa_chain.invoke({"query": USER_QUERY})
            
            print("\nAnswer:", response["result"])
            print("\nSource documents:", response["source_documents"])

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    main()