import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.core.postprocessor import FixedRecencyPostprocessor
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.llms.gemini import Gemini
from streamlit_cookies_controller import CookieController
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from typing import List, Optional
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, MetadataFilter, FilterOperator

class FixedRecencyPostprocessor(BaseNodePostprocessor):
    """Fixed Recency post-processor.

    This post-processor does the following steps orders nodes by date.

    Assumes the date_key corresponds to a date field in the metadata.
    """

    top_k: int = 1
    date_key: str = "date"

    @classmethod
    def class_name(cls) -> str:
        return "FixedRecencyPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. Please install it with pip install pandas."
            )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        # sort nodes by date
        node_dates = pd.to_datetime(
            [node.node.metadata[self.date_key] for node in nodes],
            dayfirst=True,
        )
        logging.info(f"Node dates: {node_dates}")
        sorted_node_idxs = np.flip(node_dates.argsort())
        sorted_nodes = [nodes[idx] for idx in sorted_node_idxs]

        return sorted_nodes[: self.top_k]


import os
import utils  # Importing the utilities script for link processing
from pdf2text import MyFileReader  # Importing the custom PDF reader
import requests
import shutil  # For moving files from 'new' to 'data'

# Global Variable for VectorStoreIndex
if "index" not in st.session_state:
    st.session_state.index = None

# Initialize Cookie Controller
controller = CookieController()

# Streamlit setup
st.title("DocChat")
st.markdown(
    "DocChat is a document search and chat application that uses the Llama Index library, Gemini, and Neo4j for document indexing and retrieval."
)

# Step 1: Manage API Keys and Neo4j Details with Cookies
with st.sidebar:
    st.header("Configuration")

    # Retrieve credentials from cookies if available
    api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=controller.get("api_key")
    )
    neo4j_url = st.text_input(
        "Neo4j Database URL", 
        value=controller.get("neo4j_url")
    )
    neo4j_username = st.text_input(
        "Neo4j Username", 
        value=controller.get("neo4j_username")
    )
    neo4j_password = st.text_input(
        "Neo4j Password", 
        type="password", 
        value=controller.get("neo4j_password")
    )
    embed_dim = st.number_input(
        "Embedding Dimension", 
        min_value=64, 
        max_value=1024, 
        value=768
    )

    # Save credentials in cookies
    if st.button("Save Credentials"):
        controller.set("api_key", api_key, max_age=3600)  # 1 hour
        controller.set("neo4j_url", neo4j_url, max_age=3600)
        controller.set("neo4j_username", neo4j_username, max_age=3600)
        controller.set("neo4j_password", neo4j_password, max_age=3600)
        st.success("Credentials saved in cookies.")

    # Clear cookies
    if st.button("Clear Credentials"):
        controller.remove("api_key")
        controller.remove("neo4j_url")
        controller.remove("neo4j_username")
        controller.remove("neo4j_password")
        st.success("Credentials cleared from cookies.")

# Step 2: Define paths
base_dir = "data"  # Base directory for data storage
new_dir = os.path.join(base_dir, "new")  # 'new' folder inside 'data'

# Ensure the 'new' directory exists
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Step 3: Input Document Links
st.header("Upload and Process Documents")
uploaded_links = st.text_area("Enter URLs (one per line):")
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {}
if st.button("Process Links"):
    if not api_key:
        st.error("Please provide your API Key in the sidebar!")
    else:
        with st.spinner("Processing links and downloading PDFs..."):
            links = uploaded_links.splitlines()
            pdf_links = []
            for link in links:
                try:
                    st.session_state.extracted_data.update(utils.scrape_urls(link))
                    pdf_links.extend([row['attachment'] for row in st.session_state.extracted_data.values()])
                except Exception as e:
                    st.error(f"Error processing {link}: {e}")
            
            # Initialize progress bar
            overall_progress = st.progress(0)
            total_pdfs = len(pdf_links)
            downloaded = 0
            for index, pdf_url in enumerate(pdf_links):
                pdf_name = os.path.basename(pdf_url)
                pdf_path = os.path.join(new_dir, pdf_name)
                try:
                    # Download the PDF using requests
                    response = requests.get(pdf_url, stream=True)
                    response.raise_for_status()  # Raise an error for HTTP issues
                    with open(pdf_path, "wb") as pdf_file:
                        downloaded += 1
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:  # Filter out keep-alive chunks
                                pdf_file.write(chunk)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error downloading {pdf_url}: {e}")
                
                # Update overall progress bar
                overall_progress.progress((index + 1) / total_pdfs)
            
            st.success(f"Downloaded {downloaded} PDFs!")

# Step 4: Index Documents
Settings.embed_model = GeminiEmbedding(api_key=api_key, model_name="models/text-embedding-004")
Settings.llm = Gemini(api_key=api_key, model="models/gemini-1.5-flash")
if "vector_store" not in st.session_state and all([api_key, neo4j_url, neo4j_username, neo4j_password]):
    st.session_state.vector_store = Neo4jVectorStore(
        neo4j_username,
        neo4j_password,
        neo4j_url,
        embed_dim,
        hybrid_search=True,
    )
    st.session_state.index = VectorStoreIndex.from_vector_store(vector_store=st.session_state.vector_store)
if st.button("Index Documents"):
    if not all([api_key, neo4j_url, neo4j_username, neo4j_password]):
        st.error("Please set all credentials in the sidebar.")
    else:
        with st.spinner("Indexing documents... This may take a few minutes..."):
            try:
                # Configure embeddings and Neo4j store
                # qa_extractor = QuestionsAnsweredExtractor(questions=5)
                # summary_extractor = SummaryExtractor()  
                def get_metadata(file_path):
                    return st.session_state.extracted_data.get(os.path.basename(file_path), {})
                # Load documents from the 'new' directory (where PDFs were downloaded)
                documents = SimpleDirectoryReader(input_dir=new_dir,file_metadata=get_metadata,file_extractor={".pdf": MyFileReader()}).load_data()
                storage_context = StorageContext.from_defaults(vector_store=st.session_state.vector_store)
                st.session_state.index=VectorStoreIndex.from_documents(documents,  storage_context=storage_context, show_progress=True)
                st.success("Documents indexed successfully!")

                # After indexing, move the files from 'new' to the 'data' directory and clear 'new'
                for file_name in os.listdir(new_dir):
                    old_path = os.path.join(new_dir, file_name)
                    new_path = os.path.join(base_dir, file_name)
                    shutil.move(old_path, new_path)

                # Empty 'new' folder after moving files
                for file_name in os.listdir(new_dir):
                    file_path = os.path.join(new_dir, file_name)
                    os.remove(file_path)

            except Exception as e:
                st.error(f"Error indexing documents: {e}")

# Step 5: Chat Engine
st.header("Chat with Your Documents")
query = st.text_input("Ask a question about the documents:")

if st.button("Ask"):
    if not query:
        st.error("Please enter a query.")
    else:
        try:
            # Chat engine setup

# Define metadata filters
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="type", value="Policy Documents")]
            )
            memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
            chat_engine = st.session_state.index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=( 
                    "You are an expert in Indian trade policies. Use the policy documents as the foundation for your answers. "
                ),
                filters=filters
            )
            not_equal_filter = MetadataFilter(
                key="type",
                value="Policy Documents",
                operator=FilterOperator.NE
            )
            metadata_filters = MetadataFilters(
                filters=[not_equal_filter]
            )
            # Query the engine
            response = chat_engine.chat(query)
            st.subheader("Base Answer:")
            st.write(response.response)
            source_nodes = response.source_nodes
            # Define recency postprocessor
            # recency_postprocessor = FixedRecencyPostprocessor(top_k=25, date_key="date")
            # Configure chat engine with recency postprocessor
            chat_engine = st.session_state.index.as_chat_engine(
                memory=memory,
                system_prompt=( 
                    "You are an expert in Indian trade policies and you have used policy documents as the foundation for your answers. "
                    " Use the context provided to update and refine the generated output provided below with the latest information and context."
                    "Output:\n{response}"
                ),
                filters=metadata_filters,
                # node_postprocessors=[recency_postprocessor]
            )
            # Use the chat engine
            response = chat_engine.chat(query)
            source_nodes.extend(response.source_nodes)
            # Display response
            st.subheader("Refined Answer:")
            st.write(response.response)

            # Display relevant nodes with page numbers
            st.subheader("Relevant Nodes:")
            st.write(f"Retrieved {len(source_nodes)} nodes")
            for node in source_nodes:
                st.write(f"     Document Name: {node.metadata.get('title', '')}")
                st.write(f"     Page Number: {node.metadata.get('page_label', '')}")
                st.markdown("       [Document link](%s)" % node.metadata.get("attachment", ""))
                st.write(f"- {node.node.get_text().strip()}")

        except Exception as e:
            st.error(f"Error processing query: {e}")
