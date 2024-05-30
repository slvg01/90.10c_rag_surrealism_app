# ingestion and vectorization of the content
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

# Embedding Initialization
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
embeddings = []
docs = []
source_folder_path = "database"
land_folder_path = "vector_db"  # folder to save the database
chunk_size = 750
chunk_overlap = 150


# function to Load the document with their metadata
def document_load_and_chunk(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        metadata = reader.metadata  # Extract metadata
        title = metadata.title if metadata.title else "Untitled"
        author = metadata.author if metadata.author else "Unknown"
        location = os.path.basename(file_path)

        for page in reader.pages:
            text += page.extract_text()
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = text.lower()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)

    # Include content and metadata in each chunk
    chunk_data = [
        {
            "text": chunk,
            "metadata": {"title": title, "author": author, "location": location},
        }
        for chunk in chunks
    ]
    return chunk_data


# Function to save embeddings into a FAISS index
def embed_and_store_to_db(chunks, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Extract texts and metadata separately for embeddings
    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]

    # Instantiate a FAISS index
    embeddings = embeddings_model
    index = FAISS.from_texts(texts, embeddings, metadata)
    index.save_local(folder_path)

    faiss_instance = FAISS.load_local(
        folder_path, embeddings, allow_dangerous_deserialization=True
    )
    index_size = len(faiss_instance.index_to_docstore_id)
    print(f"Total number of documents: {index_size}")
    print(f"FAISS index saved successfully in folder: {folder_path}")
    return faiss_instance




# use the function to create the chunks list of all documents 
all_chunks = []

for filename in os.listdir(source_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(source_folder_path, filename)
        chunks = document_load_and_chunk(file_path)
        all_chunks.extend(chunks)

print(all_chunks[:3])

# use the function to embed each chunk and return the vectorized database instance
faiss_instance = embed_and_store_to_db(all_chunks, land_folder_path)
print(f"Number of chunks: {len(all_chunks)}")

