from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_docs")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

prompt_template='''As an expert legal assistant, your task is to accurately summarize legal documents while maintaining clarity, conciseness, and adherence to legal precision. Follow these directives to ensure optimal summaries:  

1. **Preserve Key Legal Points**: Extract essential legal details, including case laws, statutes, obligations, and rights, without omitting critical information.  
2. **Structured Summarization**: Organize the summary under appropriate headings such as **Key Facts, Legal Issues, Rulings, and Implications** to enhance readability.  
3. **Maintain Objectivity**: Provide a neutral and unbiased summary without interpretations, assumptions, or personal opinions.  
4. **Concise Language**: Use clear, precise, and legally appropriate terminology while avoiding unnecessary verbosity.  
5. **Relevance Filtering**: Focus only on information directly related to the legal matter at hand. Exclude any extraneous or repetitive content.  
6. **Avoid Legal Advice**: Do not provide personal legal opinions or recommendations; strictly summarize the document’s content.  
7. **Standardized Formatting**: Present information in bullet points or short paragraphs for better readability.  

**Legal Document:**  
{context}  

Question: {question}

**Summary:**  
'''

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def get_response(embeddings,prompt):
  results = collection.query(
        query_embeddings=[embeddings],  # Pass query embeddings
        n_results=3  # Retrieve top 3 relevant documents
    )
  retrieved_docs = "\n\n".join(results["documents"][0])
  formatted_prompt=prompt_template.format(context=retrieved_docs,question=prompt)
  response = hf_hub_llm(formatted_prompt)
  summary_start = response.find("Summary:")  # Find where the actual summary starts
  if summary_start != -1:
        response = response[summary_start + len("Summary:"):].strip()  # Remove extra text before "Summary:"
    
  return response



hf_hub_llm = HuggingFaceHub(
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
     model_kwargs={"temperature": 0.3, "max_new_tokens":1024},
)






# from openai import OpenAI

# class LegalDocumentRAG:
#     def __init__(self, vector_db, llm_model):
#         self.vector_db = vector_db
#         self.llm = OpenAI()
    
#     def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
#         """Retrieve most relevant document chunks based on semantic similarity"""
#         query_embedding = generate_embeddings([query])[0]
#         results = self.vector_db.search(query_embedding, top_k)
#         return [result.text for result in results]
    
#     def generate_summary(self, query: str, context_chunks: List[str]) -> str:
#         """Generate a tailored summary based on specific query and retrieved context"""
#         prompt = f"""
#         You are a legal research assistant. Based on the following document context 
#         and specific query, provide a precise and legally accurate summary:
        
#         Query: {query}
#         Context Chunks: {' '.join(context_chunks)}
        
#         Summary should be:
#         - Concise and clear
#         - Highlight key legal points
#         - Address the specific information need
#         """
        
#         response = self.llm.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[{"role": "system", "content": prompt}]
#         )
        
#         return response.choices[0].message.content