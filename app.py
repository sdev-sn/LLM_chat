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
import tiktoken
import time

from chromadb import Client
from langchain.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)



dotenv.load_dotenv()
##Model
claude_api_key =st.secrets["CLAUDE_API_KEY"]
llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0, anthropic_api_key=claude_api_key)

@st.cache_resource(ttl="2h")

def pre_req(file_path):
  template_message =[SystemMessage(content="You are helpful assistant"),
                     MessagesPlaceholder(variable_name="chat_history"),
                     HumanMessagePromptTemplate.from_template("{question}")]
  prompt_template = ChatPromptTemplate.from_messages(template_message)
  #get file
  df = pd.read_excel(file_path)
  df_data = df[['title','subreddit', 'selftext']]
# st.dataframe(df_data.head())

  # Convert relevant columns to string type
  for col in ['title', 'subreddit', 'selftext']:
    if col in df_data.columns:
      df_data[col] = df_data[col].astype(str)
  documents = []
  for i, row in df_data.iterrows():
    # Assuming each row is a document
      document_text = row [2]
      documents.append(document_text)

#  documents = [row[2] for _, row in df_data.iterrows()]

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=250)
  texts = text_splitter.create_documents(documents)
  #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  persist_directory = "./chroma_db_b"
  vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
  # retriever = vectorstore.as_retriever(k=100)
  return vectorstore

file_path = "reddit_data2.xlsx"
vectorstore = pre_req(file_path)

#Strealit app
st.title("Chat with Reddit")
st.caption('Hi there! I am your co-pilot for Reddit insights')

st.sidebar.title("Chat History")

task = st.sidebar.radio("Mode", ['Chat', 'Reset'])

if task == "Chat":
  retriever = vectorstore.as_retriever(search_kwargs={"k": 200})
  memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
  template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer."
  context: {context}
  {chat_history}
  Human: {question}
  Asistant:"""
  prompt = PromptTemplate(input_variable =["context" ,"chat_history", "question"], template = template)

#create custom chain
  if llm is not None and retriever is not None:
    chain = ConversationalRetrievalChain.from_llm(memory=memory, llm=llm, retriever=retriever, return_source_documents=True, combine_docs_chain_kwargs={'prompt' : prompt })
  else:
    logger.error("LLM or retriever is not initialized.")

  task = "Reset"

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

txt = st.chat_input("Ask a question or ask for questions to chat on")

if txt:
    st.session_state['chat_history'].append("User: " +txt)
    chat_user = st.chat_message("user")
    chat_user.write(txt)
    chat_assistant = st.chat_message ("assistant")
    with st.status("Getting the answer...")as status:
        tms_start = time.time()
        pred = chain({"question": txt, "chat_history": st.session_state['chat_history']}, return_only_outputs=True)
        answer = pred['answer']
        chat_assistant.write(answer)
        st.session_state['chat_history'].append("Assistant: " +answer)
        tms_elapsed = time.time() - tms_start
      #  status.update(label="Answer generated in %0.2f seconds." \
      #                % (tms_elapsed), state="complete", expanded=True)
    st.sidebar.markdown(
        "<br />".join(st.session_state['chat_history'])+"<br /><br ?>",
        unsafe_allow_html=True
    )
