from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import chainlit as cl
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sqlalchemy import create_engine
from local_settings import postgresql as settings
# # from llama_index.core import SQLDatabase, ServiceContext, ObjectIndex, VectorStoreIndex
# # from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema
# from llama_index.core.query_engine import NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine
# from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chainlit as cl
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy_utils import database_exists, create_database
from local_settings import postgresql as settings

from llama_index.core.schema import Document
from llama_index.readers.database import DatabaseReader

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex

from llama_index.core import SQLDatabase
import textwrap

from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core.query_engine import NLSQLTableQueryEngine

from llama_index.core.objects import ObjectIndex
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema

import pandas as pd

from llama_index.core.query_engine import SQLTableRetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)

from llama_index.core.objects import ObjectIndex
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema
# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request body
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    answer: str

# Function to create a PostgreSQL engine
def get_engine(pguser, pgpass, pghost, pgport, pgdb):
    url = f"postgresql://{pguser}:{pgpass}@{pghost}:{pgport}/{pgdb}"
    engine = create_engine(url, pool_size=50, echo=False)
    return engine

# Initialize LLM and Embedding Model
llm = Ollama(model="llama3", request_timeout=120.0, streaming=True)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

# Create SQLAlchemy engine
engine = get_engine(settings['pguser'],
                    settings['pgpass'],
                    settings['pghost'],
                    settings['pgport'],
                    settings['pgdb'])

# Table details and initialization


sql_database = SQLDatabase(engine, include_tables=include_tables)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    callback_manager=callback_manager
)

# Create SQLTableRetrieverQueryEngine instance
tables = list(sql_database._all_tables)
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [SQLTableSchema(table_name=table, context_str=table_details[table]) for table in tables]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
    service_context=service_context
)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=3), service_context=service_context
)

# FastAPI endpoint to handle queries
@app.post("/query", response_model=QueryResponse)
def handle_query(query_request: QueryRequest):
    prompt = query_request.prompt
    if prompt.lower() == "list all tables":
        # Return list of tables
        return QueryResponse(answer=", ".join(sql_database._all_tables))
    else:
        # Handle other types of prompts or queries
        try:
            # Use query engine to generate response based on prompt
            response = query_engine.execute(prompt)
            return QueryResponse(answer=response)  # Adjust as per the response structure from query engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)