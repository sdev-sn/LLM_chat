import os
import sys
import pytest


def test_data_file_exists():
    assert os.path.exists("reddit_data2.xlsx"), "Reddit data file missing"


def test_requirements_file_exists():
    assert os.path.exists("requirements.txt")


def test_app_file_exists():
    assert os.path.exists("app.py")


def test_imports():
    # Verify all key packages are importable
    import streamlit
    import langchain
    import langchain_text_splitters
    import langchain_anthropic
    import langchain_community
    import chromadb
    import pandas
    import tiktoken
    import openpyxl


def test_no_syntax_errors():
    import ast
    with open("app.py", "r") as f:
        source = f.read()
    # Will raise SyntaxError if app.py has syntax issues
    ast.parse(source)


def test_critical_imports_in_app():
    # Ensure the fixed import path is used, not the old broken one
    with open("app.py", "r") as f:
        source = f.read()
    assert "from langchain_text_splitters import" in source, \
        "app.py must use langchain_text_splitters, not langchain.text_splitters"
    assert "from langchain.text_splitters import" not in source, \
        "Old langchain.text_splitters import found — this will break on Streamlit Cloud"
