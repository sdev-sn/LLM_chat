#import streamlit as st
#st.write('Hello world!')

import streamlit as st
import os
import dotenv
import uuid
import pandas as pd
import openpyxl
import ragatouille

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

#from rag_methods import (
#    load_doc_to_db, 
#    load_url_to_db,
#    stream_llm_response,
#    stream_llm_rag_response,
#)

dotenv.load_dotenv()

df = pd.read_excel("reddit_data_comments_feb22.xlsx")
st.write("success")
