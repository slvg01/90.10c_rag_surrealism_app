#ingestion and vectorization of the content
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import re




# Load generative key based on secrets file
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    print("OPENAI_API_KEY not found in secrets.")
    exit(1)

#Embedding Initialization 

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)  
embeddings = []
docs = []
source_folder_path = 'database'
land_folder_path = 'vector_db'  # folder to save the database
chunk_size = 300
chunk_overlap = 100



#load the document 
def document_load_and_chunk (file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            #text = re.sub(r'\s+', ' ', text)
            text = text.lower()

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap = chunk_overlap, 
    
)
    chunks = splitter.split_text(text)
    return chunks



# Function to save embeddings into a FAISS index
def embed_and_store_to_db(chunks, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Instantiate a FAISS index
    embeddings = embeddings_model
    index = FAISS.from_texts(chunks, embeddings)
    index.save_local(folder_path)
    
    
    faiss_instance = FAISS.load_local(land_folder_path, embeddings, allow_dangerous_deserialization=True)
    index_size = len(faiss_instance.index_to_docstore_id)
    print(f"Total number of documents: {index_size}")
   
    #print("Length of the FAISS index:", index_length)
    print(f"FAISS index saved successfully in folder: {folder_path}")
    return faiss_instance
    

# Split the page into chunks

all_chunks = []


for filename in os.listdir(source_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(source_folder_path, filename)
        chunks = document_load_and_chunk(file_path)
        all_chunks.extend(chunks)
print (all_chunks) 

#Embed each chunk with OpenAI 
faiss_instance = embed_and_store_to_db(all_chunks, land_folder_path)

print (f'nb of chunks {(len(all_chunks))}')
# query = 'andre breton, 1924'
# index  = 'vector_db/index.faiss'
# docs = index.similarity_search(query, k=3)