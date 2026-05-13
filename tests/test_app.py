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
    with open("app.py", "r") as f:
        source = f.read()
    assert "from langchain_text_splitters import" in source, \
        "app.py must use langchain_text_splitters, not langchain.text_splitters"
    assert "from langchain.text_splitters import" not in source, \
        "Old langchain.text_splitters import found — this will break on Streamlit Cloud"


def test_dataframe_row_access():
    import pandas as pd
    with open("app.py", "r") as f:
        source = f.read()
    # row[2] is a KeyError when iterrows() returns named-index Series
    assert "row[2]" not in source and "row [2]" not in source, \
        "Integer index on iterrows() row — use row['column_name'] instead"


def test_excel_columns():
    import pandas as pd
    df = pd.read_excel("reddit_data2.xlsx")
    for col in ["title", "subreddit", "selftext"]:
        assert col in df.columns, f"Expected column '{col}' not found in reddit_data2.xlsx"
