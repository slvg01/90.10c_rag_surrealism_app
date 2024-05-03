#ingestion and vectorization of the content
import os
import numpy as np
from PyPDF2 import PdfReader
import openai
import streamlit as st
import faiss
#from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# import docx



# Load generative key based on secrets file
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    print("OPENAI_API_KEY not found in secrets.")
    exit(1)


# Initialize Faiss index
dimension = 768  # Assuming OpenAI embeddings are 768-dimensional
index = faiss.IndexFlatL2(dimension)  # L2 distance is suitable for cosine similarity

# Function to split documents into chunks
def split_document(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Function to embed chunks with OpenAI
def embed_with_openai(chunk):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=chunk,
            max_tokens=50,
            temperature=0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
    except openai.OpenAIError as e:
        print(f"OpenAI API call failed: {e}")
        return None
        
    embedding = np.array(response.choices[0].embedding)
    return embedding

# Specify the folder containing PDF files
folder_path = 'database'

# Initialize lists to store embeddings and corresponding documents
embeddings = []
docs = []

# Iterate over each file in the folder
for file_name in os.listdir("database"):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)

                # Iterate over each page of the PDF
                for page_number in range(num_pages):
                    page = pdf_reader.pages[page_number]
                    text = page.extract_text()

                    # Split the page into chunks
                    chunks = split_document(text)

                    # Embed each chunk with OpenAI and add to Faiss index
                    for chunk in chunks:
                        embedding = embed_with_openai(chunk)
                        embeddings.append(embedding)
                        docs.append(chunk.encode('utf-8'))
        except (IOError, FileNotFoundError) as e:
            print(f"Failed to open or read PDF file {file_path}: {e}")
            continue        

# Convert lists to numpy arrays
embeddings = np.array(embeddings)
docs = np.array(docs)

# Create directory if it doesn't exist
vector_db_folder = 'vector_db'
os.makedirs(vector_db_folder, exist_ok=True)

# Save embeddings and docs to the folder
np.save(os.path.join(vector_db_folder, 'embeddings.npy'), embeddings)
np.save(os.path.join(vector_db_folder, 'docs.npy'), docs)


# Save the Faiss index
faiss.write_index(index, os.path.join(vector_db_folder, "vector_db.index"))