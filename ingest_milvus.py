import json
import os

from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings

# ==========================================
# Load Environment
# ==========================================

load_dotenv()

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")

INPUT_JSON = os.getenv("INPUT_JSON")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION")

# multilingual-e5-large
VECTOR_DIM = 1024

# ==========================================
# Watsonx Embedding
# ==========================================

credentials = Credentials(
    api_key=WATSONX_APIKEY,
    url=WATSONX_URL,
)

embedding_model = Embeddings(
    model_id="intfloat/multilingual-e5-large",
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
)

# ==========================================
# Milvus Connection
# ==========================================

connections.connect(
    alias="default",
    uri=MILVUS_URI,
    token=MILVUS_TOKEN if MILVUS_TOKEN else None,
)

# ==========================================
# Collection Create
# ==========================================

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        max_length=128,
        is_primary=True,
        auto_id=False,
    ),
    FieldSchema(
        name="system",
        dtype=DataType.VARCHAR,
        max_length=256,
    ),
    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=1024,
    ),
    FieldSchema(
        name="content",
        dtype=DataType.VARCHAR,
        max_length=16384,
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIM,
    ),
]

schema = CollectionSchema(
    fields,
    description="IT Tool Guide Knowledge Base"
)

collection = Collection(
    name=COLLECTION_NAME,
    schema=schema,
)

# ==========================================
# Load JSON
# ==========================================

with open(INPUT_JSON, "r", encoding="utf-8-sig") as f:
    docs = json.load(f)

ids = []
systems = []
titles = []
contents = []
vectors = []

for doc in docs:

    text = f"""
title:
{doc['title']}

content:
{doc['content']}
"""

    vector = embedding_model.embed_documents(
        texts=[text]
    )[0]

    ids.append(str(doc["id"]))
    systems.append(doc["system"])
    titles.append(doc["title"])
    contents.append(doc["content"])
    vectors.append(vector)

# ==========================================
# Insert
# ==========================================

collection.insert([
    ids,
    systems,
    titles,
    contents,
    vectors
])

collection.flush()

# ==========================================
# Index
# ==========================================

collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
)

collection.load()

print(
    f"Collection={COLLECTION_NAME} "
    f"Entities={collection.num_entities}"
)
