#Ref python file - Test_chromadb
#import streamlit as st
#st.write('Hello world!')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import dotenv
import uuid
import pandas as pd
import openpyxl

#import ragatouille

from openai import OpenAI
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
from chromadb import Client
import tiktoken
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

dotenv.load_dotenv()

df = pd.read_excel("reddit_data2.xlsx")
df_data = df[['title','subreddit', 'selftext']]
st.dataframe(df_data.head())

# Convert relevant columns to string type
for col in ['title','subreddit', 'selftext']:
    if col in df_data.columns:
        df_data[col] = df_data[col].astype(str)

#st.write("success")
# Create a list of documents from your dataframe
documents = []
for i, row in df_data.iterrows():
# Assuming each row is a document
  document_text = row [2]
  documents.append(document_text)
# return documents

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=250)
texts = text_splitter.create_documents(documents)
st.write(texts)

# Select embeddings
embeddings = OpenAIEmbeddings()

# Set directory for persistent storage
persist_directory = "./chroma_db"
# Store documents in ChromaDB
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings,
    persist_directory=persist_directory
)

# Create retriever interface
##retriever = db.as_retriever()
##retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
retriever = vectorstore.as_retriever()

##Model
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0,model_name="gpt-4o-mini")

# Create QA chain
#qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=retriever,return_source_documents=True)

##def generate_response(query):
#  result = qa({"query": query})
#  return result['result']
#  qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=retriever,return_source_documents=True)
#  return qa.run(user_query)
  
st.title("ChatGPT-like clone")
user_query = st.text_input("Ask a question :")

if user_query:
    qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=retriever,return_source_documents=True)
    response = qa(user_query)
    st.write(response)
