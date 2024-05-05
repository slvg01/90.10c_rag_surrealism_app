# load, split and vectorize a document using the langchain library (all in one) and make a query on doucment content
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
import os


openai_api_key = st.secrets["OPENAI_API_KEY"]
embeddings_model = OpenAIEmbeddings(
model="text-embedding-3-small", dimensions=128
)
embeddings = []
docs = []

def embed_text_list(chunks):
    
    
    try:
    # Call the OpenAI API to embed each text separately
        texts = [chunk.page_content for chunk in chunks]
        embs = OpenAIEmbeddings(texts)
        for emb in embs:
            embeddings.append(emb)  
    # Extract the embedding and append to the embeddings list
        for chunk in chunks:
            docs.append(chunk.encode('utf-8'))
    except Exception as e:
        print(f"Error embedding text: {chunks}")
        print(f"Error message: {e}")
    # print(len(embs), len(embs[3]))
    # print(len(docs), docs)
    return embeddings, docs
    


#load the document 
def document_load_and_chunk (document):
    loader = PyPDFLoader(document)
    data = loader.load()
    print(type(data[0]))
    

    chunk_size = 200
    chunk_overlap = 50

    # Split the pdf into chunk using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap, 
    separators=['.']
    )
    chunks = splitter.split_documents(data)
    return chunks
   

chunks = document_load_and_chunk('database/MANIFESTO_OF_SURREALISM.pdf')
embed_text_list(chunks)



