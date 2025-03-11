import os
import streamlit as st
import chromadb
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Advanced Imports
import torch
# from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('legal_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedLegalRAG:
    def __init__(
        self, 
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="meta-llama/Meta-Llama-3-8B-Instruct",
        db_path="./chroma_db",
        collection_name="legal_docs"
    ):
        """
        Initialize the Legal RAG system with advanced configurations.
        
        Args:
            embedding_model (str): Embedding model name
            llm_model (str): Language model repository ID
            db_path (str): Path for persistent vector database
            collection_name (str): ChromaDB collection name
        """
        load_dotenv()
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.error(f"Embedding model initialization failed: {e}")
            raise
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise
        
        # Text Splitter Configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=100, 
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Language Model
        try:
            self.llm = HuggingFaceHub(
                repo_id=llm_model,
                model_kwargs={
                    "temperature": 0.3, 
                    "max_new_tokens": 1024,
                    "device_map": "auto"
                }
            )
        except Exception as e:
            logger.error(f"Language model initialization failed: {e}")
            raise
        
        # Prompt Template
        self.prompt_template = PromptTemplate(
            template='''Advanced Legal Research Assistant Analysis:

                Context: {context}
                Query: {question}

                Comprehensive Analysis Requirements:
                1. Extract precise legal insights
                2. Decode complex legal terminology
                3. Provide structured, actionable information
                4. Highlight critical legal implications

                Detailed Response:
                ''',
            input_variables=["context", "question"]
        )

    
    def extract_text(self, uploaded_file) -> str:
        """
        Extract text from PDF with robust error handling.
        
        Args:
            uploaded_file: Streamlit uploaded file
        
        Returns:
            str: Extracted text or empty string
        """
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n\n".join([
                    page.extract_text() or "" 
                    for page in pdf.pages
                ])
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
            return ""
    
    
    def chunk_and_embed(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text and generate embeddings.

        Args:
            text (str): Input text

        Returns:
            List of dictionaries with chunk details
        """
        try:
            chunks = self.text_splitter.split_text(text)
            embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
            embeddings = [embedding.tolist() for embedding in embeddings]

            return [
                {
                    "chunk": chunk,
                    "embedding": embedding,
                    "id": str(hash(chunk))
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]
        except Exception as e:
            logger.error(f"Chunking and embedding error: {e}")
            return []

    
    def upload_document(self, document_data: List[Dict[str, Any]]) -> bool:
        """
        Upload document chunks to ChromaDB.
        
        Args:
            document_data (List[Dict]): List of document chunks
        
        Returns:
            bool: Upload success status
        """
        try:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._upload_chunk, 
                        chunk_data['id'], 
                        chunk_data['embedding'], 
                        chunk_data['chunk']
                    ) 
                    for chunk_data in document_data
                ]
                
                return all(future.result() for future in futures)
        except Exception as e:
            logger.error(f"Document upload error: {e}")
            return False
    
    
    def _upload_chunk(self, chunk_id: str, embedding: List[float], chunk: str) -> bool:
        """Individual chunk upload with duplicate prevention."""
        existing = self.collection.get(ids=[chunk_id])
        if not existing["ids"]:
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk]
            )
        return True
    
    def retrieve_and_summarize(self, query: str) -> str:
        """
        Retrieve relevant chunks and generate summary.
        
        Args:
            query (str): User's query
        
        Returns:
            str: Generated summary
        """
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            context = "\n\n".join(results["documents"][0])
            
            formatted_prompt = self.prompt_template.format(
                context=context, 
                question=query
            )
            
            response = self.llm(formatted_prompt)
            return response.strip()
        
        except Exception as e:
            logger.error(f"Retrieval and summarization error: {e}")
            return "I apologize, but I couldn't process your request. Please try again."

def streamlit_interface():
    """Enhanced Streamlit Interface"""
    st.set_page_config(
        page_title="LegalBot: Document Summarizer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("🏛️ LegalBot: Intelligent Document Summarizer")
    
    # Sidebar
    st.sidebar.header("🔧 System Configuration")
    
    # Main Content
    rag_system = EnhancedLegalRAG()
    if 'document_processed' not in st.session_state:
        st.session_state['document_processed'] = False
        st.session_state['document_data'] = None
    
    uploaded_file = st.file_uploader(
            "Upload Legal Document", 
            type=["pdf"],
            help="Upload a PDF legal document for summarization"
            )

    if uploaded_file and not st.session_state['document_processed']:
        with st.spinner("Processing Document..."):
            text = rag_system.extract_text(uploaded_file)
            document_data = rag_system.chunk_and_embed(text)
            rag_system.upload_document(document_data)
            st.session_state['document_data'] = document_data
            st.session_state['document_processed'] = True
        st.success("Document Processed Successfully!")
    
    # Query Section
    query = st.text_input("Enter your legal query:", placeholder="What specific information do you need?")
    
    if query:
        with st.spinner("Generating Summary..."):
            summary = rag_system.retrieve_and_summarize(query)
        
        st.markdown("### 📄 Summary")
        st.markdown(summary)

if __name__ == "__main__":
    streamlit_interface()