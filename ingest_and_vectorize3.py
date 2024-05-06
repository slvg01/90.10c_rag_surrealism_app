#ingestion and vectorization of the content
import os
import numpy as np
from PyPDF2 import PdfReader
import streamlit as st
import faiss
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader,  PyPDFLoader
# import docx



# Load generative key based on secrets file
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    print("OPENAI_API_KEY not found in secrets.")
    exit(1)

# Embedding Initialization 

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)  
embeddings = []
docs = []
source_folder_path = 'database'
land_folder_path = 'vector_db'  # folder to save the database
chunk_size = 200
chunk_overlap = 50

#load the document 
def document_load_and_chunk (source_folder_path, loader  = PyPDFLoader):
    loader = DirectoryLoader(source_folder_path, loader_cls = loader)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\\n\\n\\n","\\n\\n","\\n","."])    
    chunks = text_splitter.split_documents(pages)
    return chunks


# Function to save embeddings into a FAISS index
def embed_and_store_to_db(chunks, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Instantiate a FAISS index
    embeddings = embeddings_model
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(folder_path)
    
    
    faiss_instance = FAISS.load_local(land_folder_path, embeddings, allow_dangerous_deserialization=True)
    index_size = len(faiss_instance.index_to_docstore_id)
    print(f"Total number of documents: {index_size}")
   
    #print("Length of the FAISS index:", index_length)
    print(f"FAISS index saved successfully in folder: {folder_path}")
    return faiss_instance
    

# Split the page into chunks
chunks = document_load_and_chunk(source_folder_path, PyPDFLoader)

# Embed each chunk with OpenAI 
faiss_instance = embed_and_store_to_db(chunks, land_folder_path)
            
query = 'andre breton, 1924'
index  = 'database/index.faiss'
docs = index.similarity_search(query, k=3)