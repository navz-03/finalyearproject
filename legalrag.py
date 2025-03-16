import os
import streamlit as st
import chromadb
import logging
from typing import List, Dict, Any 
import streamlit as st
import logging
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from gradio_client import Client
from dotenv import load_dotenv
import pdfplumber

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.FileHandler('legal_rag.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LegalRAG:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='meta-llama/Meta-Llama-3-8B-Instruct',
                 db_path='./chroma_db', collection_name='legal_docs'):
        """Initialize the Legal RAG system."""
        load_dotenv()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # self.llm = HuggingFaceHub(repo_id=llm_model, model_kwargs={"temperature": 0.3, "max_new_tokens": 1024})
        self.llm =Client("KingNish/Very-Fast-Chatbot")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
    def extract_text(self, uploaded_file) -> str:
        """Extract text from PDF."""
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                return '\n\n'.join([page.extract_text() or '' for page in pdf.pages])
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    def process_document(self, text: str) -> list:
        """Split text into chunks and generate embeddings."""
        try:
            chunks = self.text_splitter.split_text(text)
            embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
            return [{"id": str(hash(chunk)), "chunk": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
        except Exception as e:
            logger.error(f"Chunking & embedding error: {e}")
            return []
    
    def upload_document(self, document_data: list) -> bool:
        """Upload document chunks to ChromaDB."""
        try:
            existing_ids = set(self.collection.get(ids=[d['id'] for d in document_data])['ids'])
            new_data = [d for d in document_data if d['id'] not in existing_ids]
            
            if new_data:
                self.collection.add(ids=[d['id'] for d in new_data],
                                    embeddings=[d['embedding'] for d in new_data],
                                    documents=[d['chunk'] for d in new_data])
                logger.info(f"Uploaded {len(new_data)} new chunks.")
            return True
        except Exception as e:
            logger.error(f"ChromaDB upload error: {e}")
            return False
    
    def retrieve_and_summarize(self, query: str) -> str:
        """Retrieve relevant chunks and generate a summary."""
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
            results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
            
            if not results["documents"]:
                return "No relevant information found."
            context = '\n\n'.join(results["documents"][0])
            template='''You are an advanced Legal Research AI Assistant. Provide a comprehensive, structured legal analysis based on the context and query below:

                ### **Context:**  
                    {context}

                ### **User Query:**  
                    {query}

                    ---

                ### **Analysis Guidelines:**  
                    1. **Examine the Context Thoroughly** → Extract key legal points.  
                    2. **Address the Legal Question Directly** → Provide a precise and legally sound response.  
                    3. **Use Professional Legal Language** → Maintain clarity, avoiding unnecessary complexity.  
                    4. **Break Down Complex Concepts** → Explain legal jargon in an understandable way.  
                    5. **Acknowledge Limitations** → If context is insufficient, clearly indicate this instead of making assumptions.
                    6. **knowledge limitations** → if any query asked outside the law ,say  I specialize in parsing and explaining legal texts. I am unable to provide information on macro nutrients or their function in the human body. If you have any further legal queries, I would be more than happy to assist you.  
                    7. **Provide a Structured Response:**  
                        - **Legal Summary:** (Brief overview)  
                        - **Key Legal Findings:** (Bullet points)  
                         - **Relevant Laws/Precedents (if applicable):**  
                        - **Implications & Next Steps:** (Possible legal actions or considerations)       
                '''
            template = template = template.format(context=context, query=query)
            response = self.llm.predict(Query=template,api_name="/predict")
            return response.strip()
        except Exception as e:
             logger.error(f"Retrieval error: {e}")
             return "Error generating summary."          

def main():
    """Streamlit Interface"""
    st.set_page_config(page_title="LegalBot", page_icon="⚖️", layout="wide")
    st.title("🏛️ LegalBot: Document Summarizer")
    
    rag_system = LegalRAG()
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Upload a legal document for analysis.")
    if uploaded_file:
        with st.spinner("Processing..."):
            text = rag_system.extract_text(uploaded_file)
            document_data = rag_system.process_document(text)
            if rag_system.upload_document(document_data):
                st.success("Document processed successfully!")
    
    query = st.text_input("Enter your legal query:", placeholder="Ask a question...")
    if query:
        with st.spinner("Generating Summary..."):
            summary = rag_system.retrieve_and_summarize(query)
        st.markdown("### 📄 Summary")
        st.markdown(summary)

if __name__ == "__main__":
    main()