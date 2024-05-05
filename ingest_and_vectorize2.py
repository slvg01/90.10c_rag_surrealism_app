#ingestion and vectorization of the content
import os
from langchain_openai import OpenAI
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import FAISS




# Load generative key based on secrets file
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    print("OPENAI_API_KEY not found in secrets.")
    exit(1)

# Embedding Initialization 
source_folder_path = 'database'
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)  
embeddings = []
docs = []
land_folder_path = 'vector_db'  # folder to save the database
chunk_size = 200
chunk_overlap = 50


def document_load_and_chunk (document):
    loader = PyPDFLoader(document)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(data)
    return chunks


# Function to embed a list of text using OpenAI's GPT-3.5 capabilities
def embed_text_list(chunks):
    try:
    # Call the OpenAI API to embed each text separately
        embs = embeddings_model.embed_documents(chunks)
        for emb in embs:
            embeddings.append(emb)  
    # Extract the embedding and append to the embeddings list
        for chunk in chunks:
            docs.append(chunk.encode('utf-8'))
    except Exception as e:
        print(f"Error embedding text: {chunks}")
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
    print(dimension)
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
    index_length = index.ntotal
    print("Length of the FAISS index:", index_length)
    return index




# Iterate over each file in the folder
for file_name in os.listdir("database"):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(source_folder_path, file_name)
        try:
            # Load and chink the PDF
            chunks = document_load_and_chunk(file_path)
                
            # Embed each chunk with OpenAI 
            #embedding = embed_text_list(chunks)
        
        except (IOError, FileNotFoundError) as e:
            print(f"Failed to open or read PDF file {file_path}: {e}")
            continue        


# Save embeddings to a FAISS index with the save_embeddings_to_faiss function
# save_embeddings_to_faiss(embeddings, land_folder_path)


# Save embeddings and docs to the folder
# docs = np.array(docs)
# np.save(os.path.join(land_folder_path, 'embeddings.npy'), embeddings)
# np.save(os.path.join(land_folder_path, 'docs.npy'), docs)

print(type(chunks))
print ("nb of chunks in chunks list", len(chunks))
# print ("nb of embeddings:", len(embeddings))
# print ('embedding vector length :' , len(embeddings[18]), len(embeddings[50]))
# print (' nb of docs: ', len(docs))
