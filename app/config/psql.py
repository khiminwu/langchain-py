import os
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import bigquery
import pg8000
from google.cloud.sql.connector import Connector

load_dotenv()

def Connect():

    # Set up a PGVector instance 
    connector = Connector()
    print("Connecting to Cloud SQL...",os.getenv("DB_INSTANCE_NAME"))
    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            os.getenv("DB_INSTANCE_NAME", ""),
            "pg8000",
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASS", ""),
            db=os.getenv("DB_NAME", ""),
        )
        return conn

    # Embedding function
    embedding_fn = VertexAIEmbeddings(
        model_name="text-embedding-005"
    )


    store = PGVector(
        connection_string="postgresql+pg8000://",
        use_jsonb=True,
        engine_args=dict(
            creator=getconn,
        ),
        embedding_function=embedding_fn,
        pre_delete_collection=True  
    )

    return store

