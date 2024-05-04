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

# Function to embed a list of text using OpenAI's GPT-3.5 capabilities
def embed_text_list(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            # Call the OpenAI API to embed each text separately
            response = openai.Embed({
                'text': chunk,
                'model': 'text-davinci-003',  # You can change the model if needed
            })
            # Extract the embedding and append to the embeddings list
            embeddings.append(response['embedding'])
            docs.append(chunk.encode('utf-8'))
        except Exception as e:
            print(f"Error embedding text: {text}")
            print(f"Error message: {e}")
    return embeddings, docs



# Function to save embeddings into a FAISS index
def save_embeddings_to_faiss(embeddings, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Instantiate a FAISS index
    dimension = embeddings_np.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)

    # Add embeddings to the index
    index.add(embeddings_np)

    # Save the index to disk
    try:
        index_path = os.path.join(folder_path, 'embeddings.index')
        faiss.write_index(index, index_path)
        print(f"FAISS index saved successfully in folder: {folder_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")




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
                    embedding = embed_text_list(chunks)
                    
        
        except (IOError, FileNotFoundError) as e:
            print(f"Failed to open or read PDF file {file_path}: {e}")
            continue        


# Define target folder name
folder_path = 'vector_db'


# Save embeddings to a FAISS index in the specified folder
save_embeddings_to_faiss(embeddings, folder_path)




#docs = np.array(docs)
# # Save embeddings and docs to the folder
# np.save(os.path.join(vector_db_folder, 'embeddings.npy'), embeddings)
# np.save(os.path.join(vector_db_folder, 'docs.npy'), docs)

