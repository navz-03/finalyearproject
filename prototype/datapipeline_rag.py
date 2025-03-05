import chromadb 
import pdfplumber
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create/load collection
collection = chroma_client.get_or_create_collection(name="legal_docs")



def data_ingester(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n\n"  # Add space between pages  
    return text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks



def data_embedding(chunks):
    emb_chunks=embeddings.embed_documents(chunks)
    return emb_chunks


def data_upload(chunks,emb_chunks):
    # Check for duplicates and upload only new chunks
    for chunk, emb_chunk in zip(chunks, emb_chunks):
        chunk_id = str(hash(chunk))
        existing_chunk = collection.get(ids=[chunk_id])
        
        if not existing_chunk["ids"]: 
            collection.add(
                ids=[chunk_id],
                embeddings=[emb_chunk],
                documents=[chunk]
            )
    return True