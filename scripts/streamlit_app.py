import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.llms.gemini import Gemini
from streamlit_cookies_controller import CookieController
import os
import utils  # Importing the utilities script for link processing
import requests
import shutil  # For moving files from 'new' to 'data'

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

            for index, pdf_url in enumerate(pdf_links):
                pdf_name = os.path.basename(pdf_url)
                pdf_path = os.path.join(new_dir, pdf_name)
                try:
                    # Download the PDF using requests
                    response = requests.get(pdf_url, stream=True)
                    response.raise_for_status()  # Raise an error for HTTP issues
                    with open(pdf_path, "wb") as pdf_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:  # Filter out keep-alive chunks
                                pdf_file.write(chunk)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error downloading {pdf_url}: {e}")
                
                # Update overall progress bar
                overall_progress.progress((index + 1) / total_pdfs)
            
            st.success(f"Downloaded {len(pdf_links)} PDFs!")

# Step 4: Index Documents
if st.button("Index Documents"):
    if not all([api_key, neo4j_url, neo4j_username, neo4j_password]):
        st.error("Please set all credentials in the sidebar.")
    else:
        with st.spinner("Indexing documents... This may take a few minutes..."):
            try:
                # Configure embeddings and Neo4j store
                Settings.embed_model = GeminiEmbedding(api_key=api_key, model_name="models/text-embedding-004")
                Settings.llm = Gemini(api_key=api_key, model="models/gemini-1.5-flash")
                neo4j_vector = Neo4jVectorStore(
                    neo4j_username,
                    neo4j_password,
                    neo4j_url,
                    embed_dim,
                    hybrid_search=True
                )

                def get_metadata(file_path):
                    return st.session_state.extracted_data.get(os.path.basename(file_path), {})
                # Load documents from the 'new' directory (where PDFs were downloaded)
                
                documents = SimpleDirectoryReader(input_dir=new_dir,file_metadata=get_metadata).load_data()
                storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
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

if st.button("Chat"):
    if not query:
        st.error("Please enter a query.")
    else:
        try:
            # Chat engine setup
            memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
            chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=( 
                    "You are a chatbot, able to have normal interactions, as well as use "
                    "the context provided to return complete, detailed answers."
                )
            )

            # Query the engine
            response = chat_engine.chat(query)

            # Display response
            st.subheader("Answer:")
            st.write(response.response_text)

            # Display relevant nodes with page numbers
            st.subheader("Relevant Nodes:")
            for node in response.source_nodes:
                st.write(f"- Page {node.metadata.get('page_label', 'Unknown')}: {node.content}")

        except Exception as e:
            st.error(f"Error processing query: {e}")
