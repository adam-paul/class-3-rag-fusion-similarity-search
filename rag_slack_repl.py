#!/usr/bin/env python3
"""
rag_slack_repl.py

A unified RAG-style REPL that:
1) Connects to a Postgres (Neon) DB to retrieve Slack-like messages
2) Embeds them in Pinecone
3) Allows command-line queries (REPL) against the vector store
4) Generates an LLM answer with relevant context
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import time

# Updated LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from pinecone import Pinecone as PineconeClient, ServerlessSpec

def load_messages_from_db():
    """
    Connect to your Neon Postgres database and retrieve messages.
    Return a list of LangChain Document objects.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL is not set in the environment. Exiting.")
        sys.exit(1)

    # Connect to the database (psycopg2)
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Query with proper table references and joins to get readable names
        cursor.execute("""
            SELECT 
                m.content,
                c.name as channel_name,
                u.name as author_name,
                m."createdAt"
            FROM "Message" m
            JOIN "Channel" c ON m."channelId" = c.id
            JOIN users u ON m."authorId" = u.id
            WHERE m.content IS NOT NULL AND m.content != ''
            ORDER BY m."createdAt" DESC
        """)
        rows = cursor.fetchall()

        documents = []
        for row in rows:
            content, channel_name, author_name, created_at = row
            # Each row becomes a Document
            metadata = {
                "channel": channel_name,
                "author": author_name,
                "timestamp": created_at.isoformat() if created_at else None
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        cursor.close()
        conn.close()

        print(f"Retrieved {len(documents)} messages from the database.")
        return documents

    except Exception as e:
        print("Error connecting to the database:", e)
        sys.exit(1)

def embed_and_store_documents(documents):
    """
    Embeds documents using OpenAIEmbeddings and stores them in Pinecone.
    Returns a Pinecone vectorstore retriever that can be used for querying.
    """
    # Load environment keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not pinecone_api_key or not pinecone_index_name or not openai_api_key:
        print("Missing required environment variables. Please check PINECONE_API_KEY, PINECONE_INDEX, and OPENAI_API_KEY are set.")
        sys.exit(1)

    # Initialize Pinecone with new class-based pattern
    pc = PineconeClient(api_key=pinecone_api_key)

    # Delete existing index if it exists
    if pinecone_index_name in pc.list_indexes().names():
        print(f"Deleting existing index {pinecone_index_name}...")
        pc.delete_index(pinecone_index_name)
        
        # Wait for index to be fully deleted
        while True:
            try:
                pc.describe_index(pinecone_index_name)
                print("Waiting for index deletion to complete...")
                time.sleep(1)
            except Exception as e:
                if "not found" in str(e).lower():
                    print("Index deletion confirmed.")
                    break
                else:
                    raise e

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )

    print(f"Creating new Pinecone index: {pinecone_index_name}")
    pc.create_index(
        name=pinecone_index_name,
        dimension=3072,  # dimension for text-embedding-3-large
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Free tier only supports us-east-1
        )
    )

    print(f"Loading {len(documents)} documents into Pinecone...")
    vectorstore = Pinecone.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=pinecone_index_name
    )
    
    print("Documents loaded into Pinecone.")
    return vectorstore.as_retriever()

def run_repl(retriever):
    """
    A simple REPL loop that:
    - Takes a user query
    - Retrieves the most relevant docs
    - Adds the doc content as context
    - Calls the LLM to get a final answer
    - Repeats until user types 'exit'
    """
    # Create a ChatOpenAI LLM wrapper
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY not set in environment. Exiting.")
        sys.exit(1)

    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    # The prompt template we can use:
    # You can customize this further, just make sure
    # to pass `query` and `context` in your final `.format()`
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="{query}\n\nContext:\n{context}"
    )

    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting REPL.")
            break

        # Retrieve the most relevant docs from Pinecone
        docs = retriever.get_relevant_documents(user_query)
        # Build a context string out of the docs
        context_snippets = []
        for i, doc in enumerate(docs, start=1):
            snippet = (
                f"--- Document {i} ---\n"
                f"Channel ID: {doc.metadata.get('channel')}\n"
                f"Author ID: {doc.metadata.get('author')}\n"
                f"Created At: {doc.metadata.get('timestamp')}\n"
                f"Content: {doc.page_content}\n"
            )
            context_snippets.append(snippet)

        context_text = "\n".join(context_snippets)
        print("\nRetrieved context:")
        print(context_text)
        print("__________________________")

        # Format the prompt
        prompt_with_context = prompt_template.format(query=user_query, context=context_text)

        # Call the LLM
        answer = llm.predict(prompt_with_context)
        print("\nLLM Answer:")
        print(answer)

def main():
    # Load .env environment variables
    load_dotenv()

    # 1) Load messages from DB
    documents = load_messages_from_db()
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2) Embed & store in Pinecone, get a retriever
    retriever = embed_and_store_documents(documents)

    # 3) Run local REPL for user queries
    run_repl(retriever)

if __name__ == "__main__":
    main()

