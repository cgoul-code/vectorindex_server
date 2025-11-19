# editor_api.py
import asyncio
from quart import Quart, request, jsonify
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
import logging, os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings
from quart_cors import cors
from azure.storage.blob import BlobServiceClient
from llama_index.core.schema import TextNode



load_dotenv(find_dotenv())

# --- KONFIG ---
INDEX_NAME = "hvaerinnafor_qa_bank"

# Lokal sti til indexen din:
PERSIST_DIR = os.path.join("blobstorage", "chatbot", INDEX_NAME)
# Hvis du har annen sti, endre her:
# PERSIST_DIR = "/path/to/your/index/folder"

print(f"Loading index from: {PERSIST_DIR}")

# --- LOAD INDEX ---
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

docstore = index.docstore
nodes = list(docstore.docs.values())

print(f"Found {len(nodes)} nodes.")

updated = 0

# --- UPDATE METADATA ---
for node in nodes:
    if node.metadata is None:
        node.metadata = {}

    # Only update if it's missing or wrong
    if node.metadata.get("valid") is not 1:
        node.metadata["valid"] = 1

    # Save change into docstore
    # allow_update=True overwrites existing
    docstore.add_documents([node], allow_update=True)

print(f"Updated {updated} nodes with metadata.valid = True")

# --- PERSIST ---
print("Persisting updated index files...")
storage_context.persist(persist_dir=PERSIST_DIR)

print("Done! All nodes now have metadata.valid = True.")