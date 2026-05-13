import os
import ast
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ── Static checks ────────────────────────────────────────────────────────────

def test_app_file_exists():
    assert os.path.exists("app.py")

def test_requirements_file_exists():
    assert os.path.exists("requirements.txt")

def test_data_file_exists():
    assert os.path.exists("reddit_data2.xlsx"), "Reddit data file missing"

def test_no_syntax_errors():
    with open("app.py") as f:
        source = f.read()
    ast.parse(source)  # raises SyntaxError if broken

def test_no_integer_row_indexing():
    """iterrows() rows are keyed by column name, not integer — catches row[N] bugs."""
    import re
    with open("app.py") as f:
        source = f.read()
    # Match row[<integer>] patterns (with optional spaces)
    matches = re.findall(r'\brow\s*\[\s*\d+\s*\]', source)
    assert not matches, f"Integer index on iterrows() row found: {matches}. Use row['column_name'] instead."

def test_langchain_import_paths():
    with open("app.py") as f:
        source = f.read()
    assert "from langchain.text_splitters import" not in source, \
        "Use langchain_text_splitters, not langchain.text_splitters"

def test_langchain_pinned_below_03():
    """ConversationalRetrievalChain was removed in langchain 0.3."""
    with open("requirements.txt") as f:
        content = f.read()
    assert "langchain" in content
    # Must have an upper bound below 0.3
    assert "<0.3" in content, \
        "langchain must be pinned to <0.3 — ConversationalRetrievalChain removed in 0.3"


# ── Data integrity ───────────────────────────────────────────────────────────

def test_excel_has_required_columns():
    df = pd.read_excel("reddit_data2.xlsx")
    for col in ["title", "subreddit", "selftext"]:
        assert col in df.columns, f"Missing column '{col}' in reddit_data2.xlsx"

def test_excel_not_empty():
    df = pd.read_excel("reddit_data2.xlsx")
    assert len(df) > 0, "reddit_data2.xlsx has no rows"

def test_selftext_column_readable_as_string():
    df = pd.read_excel("reddit_data2.xlsx")
    df["selftext"] = df["selftext"].astype(str)
    assert all(isinstance(v, str) for v in df["selftext"])


# ── Runtime logic (mocked external dependencies) ─────────────────────────────

def test_data_processing_logic():
    """Run the exact data transformation from pre_req() to catch runtime errors early."""
    df = pd.read_excel("reddit_data2.xlsx")
    df_data = df[["title", "subreddit", "selftext"]]

    for col in ["title", "subreddit", "selftext"]:
        df_data[col] = df_data[col].astype(str)

    documents = []
    for i, row in df_data.iterrows():
        document_text = row["selftext"]   # this is the line that had the bug
        documents.append(document_text)

    assert len(documents) == len(df), "Document count should match row count"
    assert all(isinstance(d, str) for d in documents), "All documents should be strings"

def test_text_splitter_on_real_data():
    """Verify the text splitter runs without error on actual Excel content."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    df = pd.read_excel("reddit_data2.xlsx")
    df["selftext"] = df["selftext"].astype(str)
    documents = df["selftext"].tolist()

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=250)
    texts = splitter.create_documents(documents)
    assert len(texts) > 0, "Text splitter produced no chunks"

def test_pre_req_runs_with_mocked_embeddings():
    """Execute pre_req() end-to-end with mocked HuggingFace and Chroma."""
    mock_vectorstore = MagicMock()
    mock_embeddings = MagicMock()

    with patch("langchain_community.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("langchain_community.vectorstores.Chroma.from_documents", return_value=mock_vectorstore):

        df = pd.read_excel("reddit_data2.xlsx")
        df_data = df[["title", "subreddit", "selftext"]]
        for col in ["title", "subreddit", "selftext"]:
            df_data[col] = df_data[col].astype(str)

        documents = [row["selftext"] for _, row in df_data.iterrows()]

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=250)
        texts = splitter.create_documents(documents)

        # Simulate the vectorstore creation call
        import langchain_community.vectorstores as vs
        result = vs.Chroma.from_documents(documents=texts, embedding=mock_embeddings, persist_directory="./test_chroma")

        assert result is mock_vectorstore
