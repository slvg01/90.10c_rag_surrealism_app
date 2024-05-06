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
chunk_size = 1000
chunk_overlap = 100


loader = DirectoryLoader(source_folder_path, loader_cls = PyPDFLoader )
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\\n\\n\\n","\\n\\n","\\n","."])    
chunks = text_splitter.split_documents(pages)


if not os.path.exists(land_folder_path):
    os.makedirs(land_folder_path)

# Instantiate a FAISS index
embeddings = embeddings_model
index = FAISS.from_documents(chunks, embeddings)
index.save_local(land_folder_path)
    
            
# query = 'the imbeciles'
# docs = index.similarity_search(query, k=3)
# print(docs)