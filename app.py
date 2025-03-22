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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
from chromadb import Client
import tiktoken
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
#from langchain.schema import HumanMessage, AIMessage
#from ragatouille import RAGPretrainedModel

#RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

#from rag_methods import (
#    load_doc_to_db, 
#    load_url_to_db,
#    stream_llm_response,
#    stream_llm_rag_response,
#)

dotenv.load_dotenv()

df = pd.read_excel("reddit_data_comments_feb22.xlsx")
st.dataframe(df.head())
#st.write("success")
# Create a list of documents from your dataframe
documents = []
for index, row in df.iterrows():
# Assuming each row is a document
  document_text = row.to_string()
  documents.append(document_text)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
texts = text_splitter.create_documents(documents)

# Select embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store from documents
db = Chroma.from_documents(texts, embeddings)

# Create retriever interface
retriever = db.as_retriever()

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(api_key=st.secrets["OPENAI_API_KEY"]), model_name="gpt-4o-mini",temperature=0,chain_type='stuff', retriever=retriever)
 
st.title("ChatGPT-like clone")
# Setup the input textfield to take questions from user
#query_text = st.text_input('Question ', placeholder='Please provide your question here.')
user_query = st.text_input("Ask a question about your Excel data:")

if user_query:
    response = qa.run(user_query)
    st.write(response)

# Set OpenAI API key from Streamlit secrets
##client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
##if "openai_model" not in st.session_state:
##    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize chat history
##if "messages" not in st.session_state:
##    st.session_state.messages = []

# Display chat messages from history on app rerun
##for message in st.session_state.messages:
##    with st.chat_message(message["role"]):
##        st.markdown(message["content"])

# Accept user input
##if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
##    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
##    with st.chat_message("user"):
 ##       st.markdown(prompt)

# Display assistant response in chat message container
 ##   with st.chat_message("assistant"):
##        stream = client.chat.completions.create(
 ##           model=st.session_state["openai_model"],
 ##           messages=[
 ##               {"role": m["role"], "content": m["content"]}
  ##              for m in st.session_state.messages
##            ],
  ##          stream=True,
   ##     )
  ##      response = st.write_stream(stream)
##    st.session_state.messages.append({"role": "assistant", "content": response})
