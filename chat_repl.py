#!/usr/bin/env python3

import os
import sys
import psycopg2  # or use another library such as SQLAlchemy
from dotenv import load_dotenv

# If using the "langchain_community" libraries, ensure they're installed:
#   pip install git+https://github.com/hwchase17/langchain
#   pip install langchain_community
#   pip install pinecone-client
#   etc.

# ---- LangChain & Pinecone imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts.prompt import PromptTemplate

# -------------------------------------------
# 0. Load environment variables
# -------------------------------------------
load_dotenv()  # This will load from .env if not in Docker

# Required environment variables - will raise error if not set
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
DATABASE_URL = os.environ["DATABASE_URL"]  # New single connection string

# Optional environment variables with defaults
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "MyLangChainProject")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# -------------------------------------------
# 1. Fetch messages from the Postgres database
# -------------------------------------------
def fetch_messages_from_db():
    """
    Fetch messages from the database using the Prisma schema structure.
    Returns a list of (content, channel_name, author_name, timestamp).
    """
    try:
        print(f"Attempting to connect to database...")
        conn = psycopg2.connect(DATABASE_URL)
        print("Successfully connected to database")
        
        # Test the connection with a simple query
        cursor = conn.cursor()
        cursor.execute('SELECT current_database(), current_user;')
        db, user = cursor.fetchone()
        print(f"Connected to database: {db} as user: {user}")
        
        # List all tables to debug
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print("Available tables:", [t[0] for t in tables])
        
        # Join with channels and users to get the readable names
        query = """
            SELECT 
                m.content,
                c.name as channel_name,
                u.name as author_name,
                m."createdAt"
            FROM "Message" m
            JOIN "Channel" c ON m."channelId" = c.id
            JOIN users u ON m."authorId" = u.id
            ORDER BY m."createdAt" DESC
        """
        print(f"Executing query: {query}")
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"Retrieved {len(rows)} messages from database")
        if len(rows) == 0:
            print("WARNING: No messages found in database!")
        else:
            print(f"Sample message: {rows[0]}")
        return rows

    except Exception as e:
        print(f"[ERROR] Could not fetch messages: {e}")
        print(f"Database URL format (censored): {DATABASE_URL.split('@')[0].split(':')[0]}:***@{DATABASE_URL.split('@')[1]}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

# -------------------------------------------
# 2. Convert DB rows into LangChain Documents
# -------------------------------------------
def create_documents_from_messages(rows):
    """
    Convert each message row into a LangChain Document object.
    We store channel, author, timestamp in the metadata dict.
    """
    print(f"Converting {len(rows)} messages to documents")
    documents = []
    for (content, channel_name, author_name, created_at) in rows:
        # Handle potential NULL values gracefully
        text_content = content if content else ""
        if not text_content.strip():
            print(f"WARNING: Empty content for message in channel {channel_name} by {author_name}")
            continue
        meta = {
            "channel": channel_name,
            "author": author_name,
            "timestamp": created_at.isoformat() if created_at else None
        }
        # Each message is a single Document
        documents.append(Document(page_content=text_content, metadata=meta))
    print(f"Created {len(documents)} documents (filtered out {len(rows) - len(documents)} empty messages)")
    return documents

# -------------------------------------------
# 3. Split documents into chunks (if desired)
# -------------------------------------------
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked = text_splitter.split_documents(documents)
    print(f"Split into {len(chunked)} chunks")
    return chunked

# -------------------------------------------
# 4. Create or load a Pinecone vector store
# -------------------------------------------
def create_or_load_vectorstore(documents):
    """
    1) Create embeddings
    2) Initialize Pinecone
    3) Upload documents to Pinecone
    4) Return a PineconeVectorStore object
    """
    from pinecone import Pinecone, ServerlessSpec
    import time

    # Using the same embedding model as in your original code
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize Pinecone with new class-based pattern
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Delete the index if it exists
    if PINECONE_INDEX in pc.list_indexes().names():
        print(f"Deleting existing index {PINECONE_INDEX}...")
        pc.delete_index(PINECONE_INDEX)
        
        # Wait for index to be fully deleted
        while True:
            try:
                pc.describe_index(PINECONE_INDEX)
                print("Waiting for index deletion to complete...")
                time.sleep(1)
            except Exception as e:
                if "not found" in str(e).lower():
                    print("Index deletion confirmed.")
                    break
                else:
                    raise e

    print(f"Creating new index {PINECONE_INDEX}...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=3072,  # dimension for text-embedding-3-large is 3072
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Free tier only supports us-east-1
        )
    )

    print(f"Creating Pinecone vectorstore from {len(documents)} documents...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX
    )
    print("Loading to vectorstore done.")
    return vectorstore

# -------------------------------------------
# 5. REPL for queries
# -------------------------------------------
def repl_loop(vectorstore):
    """
    Command-line REPL to query the vectorstore. 
    Mimics the flow from your original main.py.
    """
    # We want to replicate how main.py used the LLM
    # Specifically: retrieval, then adding the retrieved docs to context, then LLM.
    
    # We should create a retriever from the vectorstore
    retriever = vectorstore.as_retriever()
    
    # The ChatOpenAI model you used:
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

    # We also have a PromptTemplate, though you can customize it as you wish
    # or keep it exactly as in your original main.py:
    template = PromptTemplate(
        template="{query} Context: {context}",
        input_variables=["query", "context"]
    )

    print("Entering REPL mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("User> ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # 1) Retrieve context from the vectorstore
        docs = retriever.invoke(user_input)

        # Print out the retrieved docs for debugging/inspection
        print("Retrieved Context:")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("channel", "unknown_channel")
            author = doc.metadata.get("author", "unknown_author")
            timestamp = doc.metadata.get("timestamp", "unknown_timestamp")
            print(f"--- Doc {i+1} ---")
            print(f"Channel: {source}, Author: {author}, Timestamp: {timestamp}")
            print(f"Content: {doc.page_content[:250]} ...\n")  # limit to 250 chars for brevity
        
        # 2) Format the final prompt for the LLM
        prompt_with_context = template.invoke({
            "query": user_input,
            "context": docs
        })

        # 3) Query the LLM
        results = llm.invoke(prompt_with_context)

        # 4) Print out the answer
        print("Answer:\n", results.content, "\n")

# -------------------------------------------
# 6. Main execution
# -------------------------------------------
def main():
    # 1) Fetch raw messages from DB
    print("Fetching messages from the database...")
    rows = fetch_messages_from_db()

    # 2) Convert to Documents
    documents = create_documents_from_messages(rows)

    # 3) (Optional) Split them into smaller chunks
    print("Splitting messages into chunks...")
    chunked_docs = split_documents(documents)

    # 4) Create/Load VectorStore
    vectorstore = create_or_load_vectorstore(chunked_docs)

    # 5) Start REPL for user queries
    repl_loop(vectorstore)

if __name__ == "__main__":
    main()

