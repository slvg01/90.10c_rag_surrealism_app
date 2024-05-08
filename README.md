
# RAG Notion App Project: 

## 🎯 Objectives:
- create a online Retrieved Augmented Generation `RAG chat bot` app to answer questions regarding surrealism and especially belgium surrealism GET IN AND TRY](https://surrealism.streamlit.app/)
    

- The corpus of document is a set of publically available pdf documents on surrealism. Please contact me to get them if interested. 

- The LLM used in the backgroud to formulate the answer is openai 

<p align="center">
  <img src="pictures/rag_scheme.png"  />
</p>


## 🔧 Installation

To install the app scripts , follow these steps:

- Clone the repository to your local machine using the command :
    - `git clone https://github.com/slvg01/10c_rag_surrealism_app.git`
    
- Get into the project folder: 
    - `cd into 10c_rag_surrealism_app`
    
- Ensure that you have the required dependencies installed by executing:
    - `pip install -r requirements.txt`

- Set up a secret.toml file within a .streamlit folder. In the secret file save your : 
    - `OPENAI_API_KEY = 'sk-xxxxxxxxxxxxxxxxxxxxx'`

- create a database folder. Input your own pdf file about surrealism or other subject or contact me to get the files i used. 
    


## 👟 Running
- you may just try to press the `Enter` button above and try the app online, 

- Or if you are trying to duplicate or create your own app based on your own set of pdf : 
    - once the above installation is done and your ppdf are in the databse folder, then run the ingest_and_vectorized.py script to create the database index
    - run streamlit from your terminal to launch the app locally:
    streamlit run "absolute_path_to_your_streamlit_app.py"


## Credit 

To ***`logan Vendrix`*** and his [Article](https://blog.streamlit.io/build-your-own-notion-chatbot/) that pointing me in the right **direction** to do this RAG project


