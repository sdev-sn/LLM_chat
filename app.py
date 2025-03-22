#import streamlit as st
#st.write('Hello world!')

import streamlit as st
import os
import dotenv
import uuid
import pandas as pd
import openpyxl
#import ragatouille

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
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

st.title("ChatGPT-like clone")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
