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

# Check if running in Azure Web App and set the correct local storage path
if os.getenv("WEBSITE_SITE_NAME"):  # This environment variable exists in Azure App Services
    LOCAL_STORAGE_PATH = "/home/vector-index"  # Linux
else:
    LOCAL_STORAGE_PATH = "vector-index"  # Default for local testing
    
IN_AZURE = "WEBSITE_SITE_NAME" in os.environ or "FUNCTIONS_WORKER_RUNTIME" in os.environ
#IN_AZURE = True
    
# Logical name of the index (used as prefix in blob container and local folder name)
INDEX_NAME = os.getenv("INDEX_NAME", "hvaerinnafor_qa_bank")  
    
# Local directory where LlamaIndex will read/write the persistent index files
if IN_AZURE:
    # In Azure: store the index under /home/vector-index/<INDEX_NAME>
    PERSIST_DIR = os.path.join(LOCAL_STORAGE_PATH, INDEX_NAME)
else:
    # Locally: match what you already use
    # Example from your logs: blobstorage/chatbot/hvaerinnafor_qa_bank
    PERSIST_DIR = os.path.join("blobstorage", "chatbot", INDEX_NAME) 
    
# Azure blob config (only used when IN_AZURE=True)
AZ_CONNECTION_STRING = os.getenv("CONNECTION_STRING")
AZ_CONTAINER_NAME = os.getenv("CONTAINER_NAME")  # e.g. "vector-indexes"
    
    


LLMGPT4 = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    model_kwargs={"response_format": {"type": "json_object"}},
    #temperature=0.0,
    timeout=120,
)

Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
)

#download_and_persist_storage("hvaerinnafor", "./blobstorage/chatbot/hvaerinnafor" )

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True)

# -------------------------------------------------------------------
# Azure blob helpers
# -------------------------------------------------------------------

def get_container_client():
    if not (AZ_CONNECTION_STRING and AZ_CONTAINER_NAME):
        raise RuntimeError(
            "Azure CONNECTION_STRING or CONTAINER_NAME not set, "
            "but IN_AZURE is True."
        )
    blob_service_client = BlobServiceClient.from_connection_string(AZ_CONNECTION_STRING)
    return blob_service_client.get_container_client(AZ_CONTAINER_NAME)


def download_index_from_blob():
    """
    Download all blobs under prefix '<INDEX_NAME>/' into PERSIST_DIR
    (docstore.json, index_store.json, vector_store.json, etc.).
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)
    container_client = get_container_client()

    prefix = f"{INDEX_NAME}/"
    logging.info(f"[Azure] Downloading index from container '{AZ_CONTAINER_NAME}', prefix '{prefix}' to '{PERSIST_DIR}'")

    blobs = list(container_client.list_blobs(name_starts_with=prefix))
    if not blobs:
        logging.warning(f"[Azure] No blobs found with prefix {prefix}. "
                        f"Index might not exist yet in container.")

    for blob in blobs:
        blob_name = blob.name  # e.g. "hvaerinnafor_qa_bank/docstore.json"
        filename = os.path.basename(blob_name)
        local_path = os.path.join(PERSIST_DIR, filename)

        logging.info(f"[Azure] Downloading blob '{blob_name}' -> '{local_path}'")
        with open(local_path, "wb") as f:
            downloader = container_client.download_blob(blob_name)
            f.write(downloader.readall())

    logging.info("[Azure] Download complete.")


def upload_index_to_blob():
    """
    Upload all files in PERSIST_DIR back to the Azure Blob container
    under prefix '<INDEX_NAME>/'.
    """
    if not IN_AZURE:
        return

    container_client = get_container_client()
    os.makedirs(PERSIST_DIR, exist_ok=True)

    prefix = f"{INDEX_NAME}/"
    logging.info(f"[Azure] Uploading index files from '{PERSIST_DIR}' to container '{AZ_CONTAINER_NAME}', prefix '{prefix}'")

    for filename in os.listdir(PERSIST_DIR):
        local_path = os.path.join(PERSIST_DIR, filename)
        if not os.path.isfile(local_path):
            continue

        blob_name = f"{prefix}{filename}"  # e.g. "hvaerinnafor_qa_bank/docstore.json"
        logging.info(f"[Azure] Uploading '{local_path}' -> '{blob_name}'")
        blob_client = container_client.get_blob_client(blob=blob_name)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    logging.info("[Azure] Upload complete.")
    
# -------------------------------------------------------------------
# LlamaIndex init
# -------------------------------------------------------------------

def init_index():
    """Initialize global `index` and `storage_context`."""
    global index, storage_context

    if IN_AZURE:
        # In Azure: ensure we have a local copy of the index from Blob
        download_index_from_blob()
    else:
        # Locally: nothing extra, index files should already be in PERSIST_DIR
        if not os.path.isdir(PERSIST_DIR):
            raise RuntimeError(f"PERSIST_DIR does not exist locally: {PERSIST_DIR}")

    logging.info(f"Loading index from PERSIST_DIR='{PERSIST_DIR}'")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded.")
    
def node_to_dict(node):
    return {
        "node_id": node.node_id,
        "text": node.text,
        "metadata": node.metadata or {},
    }



app = Quart(__name__)
#app = cors(app, allow_origin="http://localhost:3000")  
app = cors(app, allow_origin="*")  # Allow CORS from all origins

index = None
storage_context = None


# Run on import
init_index()


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------

@app.get("/nodes")
async def list_nodes():
    """
    List nodes with simple paging and optional free-text search on node.text.
    """
    q = request.args.get("q", "").strip()
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 20))

    # Always fetch docstore from index to keep it fresh
    docstore = index.docstore
    all_nodes = list(docstore.docs.values())

    if q:
        q_lower = q.lower()
        all_nodes = [
            n for n in all_nodes
            if q_lower in (n.text or "").lower()
        ]

    total = len(all_nodes)
    page = all_nodes[offset:offset + limit]

    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [node_to_dict(n) for n in page],
    })


@app.get("/nodes/<node_id>")
async def get_node(node_id: str):
    docstore = index.docstore
    node = docstore.docs.get(node_id)
    if node is None:
        return jsonify({"error": "Node not found"}), 404
    return jsonify(node_to_dict(node))


@app.put("/nodes/<node_id>")
async def update_node(node_id: str):
    """
    Update a single node:
      - Update text + metadata in docstore
      - Re-embed & upsert only this node into vector index
      - Persist
      - If in Azure: upload updated files to Blob
    """
    global index, storage_context

    payload = await request.get_json()
    new_text = payload.get("text")
    new_metadata = payload.get("metadata", {})

    docstore = index.docstore
    node = docstore.docs.get(node_id)
    if node is None:
        return jsonify({"error": "Node not found"}), 404

    # 1) Update text + metadata
    if new_text is not None:
        node.text = new_text

    node.metadata = {**(node.metadata or {}), **new_metadata}

    # 2) Update in docstore
    # `allow_update=True` ensures we overwrite existing
    docstore.add_documents([node], allow_update=True)

    # 3) Update only this node in the vector index
    try:
        if hasattr(index, "delete_nodes"):
            index.delete_nodes([node_id])
        else:
            logging.warning("index.delete_nodes not available; skipping delete step.")
    except Exception as e:
        logging.warning(f"Could not delete node {node_id} from index: {e}")

    # Insert node again → triggers new embedding and upsert into vector store
    index.insert_nodes([node])

    # 4) Persist to local storage
    storage_context.persist(persist_dir=PERSIST_DIR)
    logging.info(f"Persisted updated index to {PERSIST_DIR}")

    # 5) If running in Azure, push updated files back to Blob
    if IN_AZURE:
        upload_index_to_blob()

    return jsonify({"status": "ok", "node": node_to_dict(node)})

@app.route("/nodes", methods=["POST"])
async def create_node():
    """
    Create a new node:
      - Create TextNode with text + metadata
      - Insert into index (docstore + vector store)
      - Persist
      - If in Azure: upload to Blob
    """
    global index, storage_context

    try:
        payload = await request.get_json()
        logging.info(f"POST /nodes payload: {payload!r}")
    except Exception as e:
        logging.exception("Failed to parse JSON payload for /nodes")
        return jsonify({"error": "Invalid JSON payload", "detail": str(e)}), 400

    text = (payload or {}).get("text", "")
    metadata = (payload or {}).get("metadata", {}) or {}

    try:
        # 1) Create the node
        node = TextNode(text=text, metadata=metadata)

        # 2) Insert into index (handles docstore + vector store)
        index.insert_nodes([node])

        # 3) Persist to local storage
        storage_context.persist(persist_dir=PERSIST_DIR)
        logging.info(f"Persisted new node to {PERSIST_DIR}")

        # 4) If running in Azure, push updated files back to Blob
        if IN_AZURE:
            upload_index_to_blob()

        return jsonify(node_to_dict(node)), 201

    except Exception as e:
        logging.exception("Error while creating node")
        return jsonify({"error": "Server error while creating node", "detail": str(e)}), 500


if __name__ == "__main__":
    # Kjør med: python editor_api.py
    app.run(host="0.0.0.0", port=8001, debug=True)
