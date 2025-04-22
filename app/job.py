import os
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings

from google.cloud import bigquery
import pg8000
from google.cloud.sql.connector import Connector
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# load_dotenv()
from app.config.psql import Connect

store = Connect()

def save_pdf_to_pgvector(file_path: str,type: str = ""):
    print("Saving PDF to PGVector...",file_path)
    
    if type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif type == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    store.add_documents(chunks)
    return full_text

# # Retrieve all Cloud Run release notes from BigQuery 
# client = bigquery.Client()
# query = """
# SELECT
#   CONCAT(FORMAT_DATE("%B %d, %Y", published_at), ": ", description) AS release_note
# FROM `bigquery-public-data.google_cloud_release_notes.release_notes`
# WHERE product_name= "Cloud Run"
# ORDER BY published_at DESC
# """
# rows = client.query(query)

# print(f"Number of release notes retrieved: {rows.result().total_rows}")


# # Set up a PGVector instance 
# connector = Connector()
# print("Connecting to Cloud SQL...",os.getenv("DB_INSTANCE_NAME"))
# def getconn() -> pg8000.dbapi.Connection:
#     conn: pg8000.dbapi.Connection = connector.connect(
#         os.getenv("DB_INSTANCE_NAME", ""),
#         "pg8000",
#         user=os.getenv("DB_USER", ""),
#         password=os.getenv("DB_PASS", ""),
#         db=os.getenv("DB_NAME", ""),
#     )
#     return conn

# # Embedding function
# embedding_fn = VertexAIEmbeddings(
#     model_name="text-embedding-005"
# )


# store = PGVector(
#     connection_string="postgresql+pg8000://",
#     use_jsonb=True,
#     engine_args=dict(
#         creator=getconn,
#     ),
#     embedding_function=embedding_fn,
#     pre_delete_collection=True  
# )

# texts = list(row["release_note"] for row in rows)
# ids = store.add_texts(texts)

# print(f"Done saving: {len(ids)} release notes")

