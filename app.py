echo "# LLM_chat" >> README.md
git init
git add README.md
git commit -m "Initial commit of Streamlit app"
git branch -M main
git remote add origin https://github.com/sdev-sn/LLM_chat.git
git push -u origin main


import streamlit as st
st.write('Hello world!')
