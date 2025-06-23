from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# load the documents
DATA_PATH = "data/"
def load_pdf_files(data):
    try:
        print(f"Loading PDFs from directory: {data}")
        if not os.path.exists(data):
            raise FileNotFoundError(f"Directory {data} does not exist")
            
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            print("Warning: No PDF files found in the directory")
            return []
            
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        print(f"Error loading PDF files: {str(e)}")
        return []

# create chunks    
def create_chunks(extracted_data):
    try:
        print("Creating chunks from documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        text_chunks = text_splitter.split_documents(extracted_data)
        print(f"Created {len(text_chunks)} chunks")
        return text_chunks
    except Exception as e:
        print(f"Error creating chunks: {str(e)}")
        return []

# create vector store
def get_embeddings_model():
    try:
        print("Initializing embeddings model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embeddings model initialized successfully")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embeddings model: {str(e)}")
        return None

def main():
    try:
        # Load PDF documents
        documents = load_pdf_files(data=DATA_PATH)
        
        if not documents:
            print("No documents were loaded. Please check if PDF files exist in the data directory.")
            return
            
        # Create chunks from the documents
        chunks = create_chunks(documents)
        
        if not chunks:
            print("No chunks were created from the documents.")
            return
            
        # Get embeddings model
        embedding_model = get_embeddings_model()
        
        if not embedding_model:
            print("Failed to initialize embeddings model.")
            return
            
        # Store embeddings in FAISS
        print("Creating FAISS vector store...")
        DB_FAISS_PATH = "vectorstore/db_faiss"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        # Create and save the vector store
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print(f"Successfully created and saved vector store at {DB_FAISS_PATH}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()