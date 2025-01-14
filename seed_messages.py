#!/usr/bin/env python3
# seed_messages.py

import os
import sys
import json
import random
import datetime
from openai import OpenAI
import psycopg2
from dotenv import load_dotenv

# 1. Load environment variables (including your OPENAI_API_KEY and DB connection URL)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY not found in environment.")
    sys.exit(1)

if not DATABASE_URL:
    print("[ERROR] DATABASE_URL (or POSTGRES_PRISMA_URL) not found in environment.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# 2. Define your four characters and any relevant context about them
CHARACTERS = {
    "crumbqueen": {
        "id": "user_crumbqueen123456789",
        "description": (
            "Insufferable red scare podcast dilettante, she is cynical about everything and pretends to know a lot about art and psychology but is always saying cringe things everyone already knows as if it makes her cool"
        )
    },
    "hyperb0re4n": {
        "id": "user_hyperborean123456789",
        "description": (
            "Absolute fitness bro, doesn't care about anything except being jacked. "
            "Constantly proclaiming to be 'natty bro' and convince people who did not ask"
        )
    },
    "notspook_normalguy": {
        "id": "user_notspook123456789",
        "description": (
            "very normal guy, always trying to seem as normal and chill as possible, then very rarely asking hyperspecific technical questions about people related to possible weapons and/or illegal cybersecurity infrastructure they might possess"
        )
    },
    "joe66": {
        "id": "user_joe66123456789",
        "description": (
            "58-year-old plumber, believes he's on a tech support forum. "
            "Often complains or asks about plumbing or home repair in a tech context."
        )
    },
}

# 3. Prompt engineering: We'll ask GPT-4o (or GPT-4-like) to generate a single JSON array
#    containing 80 messages total. Each message includes:
#    - id (any unique string or integer)
#    - character (which of the 4)
#    - content (up to 280 characters)
#    - date (some date/time between Nov 1 and Nov 30, 2024)
#    - replyToId (optional; if it's a reply, references the 'id' of a prior message)
#    - channelName (we’ll assume a single channel or a small set of channels)
SYSTEM_PROMPT = """You are a helpful assistant that will generate a JSON array of Slack-like messages from four fictional users. 
We have four characters:
1) crumbqueen (description: Insufferable red scare podcast dilettante, she is cynical about everything and pretends to know a lot about art and psychology but is always saying cringe things everyone already knows as if it makes her cool.)
2) hyperb0re4n (description: Absolute fitness bro, doesn't care about anything except being jacked.Constantly proclaiming to be 'natty bro' and convince people who did not ask)
3) notspook_normalguy (description: very normal guy, always trying to seem as normal and chill as possible, then very rarely asking hyperspecific technical questions about people related to possible weapons and/or illegal cybersecurity infrastructure they might possess)
4) joe66 (description: 58-year-old plumber, believes he's on a tech support forum. Often complains or asks about plumbing or home repair in a tech context.)

Constraints:
- Generate 80 messages total, distributed among these four characters (~20 each).
- Each message up to 280 characters in length (like a short post).
- The messages are in chronological order, from November 1 to November 30, 2024.
- Some messages are replies to others (use a 'replyToId' to reference a previous message's 'id' within this list). 
- The final output must be valid JSON (parseable) and must be an array of length 80.
- Each array element should be an object with:
    "id": a unique string or integer,
    "character": one of the four above,
    "content": the message text (<= 280 chars),
    "date": an ISO8601-like date-time between 2024-11-01 and 2024-11-30,
    "replyToId": optional, referencing the 'id' of a previous message if this is a reply,
    "channelName": any short name to indicate which channel they're in (like "general", "random", or "offtopic").

Aim for roughly 20 messages per character. Make the messages somewhat interwoven (they comment/reply to each other occasionally).
Return only valid JSON, with no additional commentary.
"""

def generate_messages_via_openai():
    """
    Calls OpenAI's API to generate a JSON array (list) of 80 messages
    in the specified format. We'll parse and return it as a Python list of dicts.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or your specific model
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        temperature=0.7,
        max_tokens=8000,  # Increased token limit to ensure we get complete responses
        response_format={ "type": "json_object" }  # Ensure we get valid JSON back
    )

    # The assistant's reply should be JSON. We parse it:
    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        # Ensure we're getting the messages array from the JSON object
        messages = data.get('messages', data)
        if not isinstance(messages, list):
            print("[ERROR] GPT response is not a JSON array. Check the prompt/response.")
            sys.exit(1)
        return messages
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse the JSON from GPT: {str(e)}")
        print("Raw output was:\n", content)
        sys.exit(1)

# 4. Insert logic: 
#    We'll do a minimal approach with psycopg2 to insert into your Prisma-based schema.
#    Typically, you'd also need to ensure that each user and channel exist. 
#    We can do a simple upsert approach (insert if not exists) for user and channel.
def get_or_create_user(conn, character_name, character_data):
    """
    Ensures a user row exists. If not, create it.
    Uses predefined Clerk-like IDs from the CHARACTERS dictionary.
    """
    user_id = character_data["id"]
    
    with conn.cursor() as cur:
        # Attempt to find existing user by id
        cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        if row:
            return row[0]
        
        # Create with predetermined ID
        now = datetime.datetime.now()
        cur.execute("""
            INSERT INTO users (id, name, status, "createdAt", "updatedAt") 
            VALUES (%s, %s, 'offline', %s, %s)
            RETURNING id
        """, (user_id, character_name, now, now))
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id

def get_or_create_channel(conn, channel_name, creator_id):
    """
    Ensures a channel row exists. If not, create it.
    Returns the channel's id. We'll store the channel_name as 'name' and do a random cuid() for the actual primary key.
    """
    with conn.cursor() as cur:
        # Attempt to find existing channel by name
        cur.execute('SELECT id FROM "Channel" WHERE name = %s', (channel_name,))
        row = cur.fetchone()
        if row:
            return row[0]
        
        # Otherwise, create a new channel
        # We'll auto-generate an ID using a random GUID. Here we'll rely on Postgres function or a random approach:
        import uuid
        channel_id = str(uuid.uuid4())
        now = datetime.datetime.now()

        cur.execute("""
            INSERT INTO "Channel" (id, name, description, "creatorId", "createdAt", "updatedAt")
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (channel_id, channel_name, f"Seeded channel: {channel_name}", creator_id, now, now))
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id

def insert_message(conn, message_data, user_id_map, channel_id_map):
    """
    Inserts a single message into the DB. 
    message_data is a dict with "id", "character", "content", "date", "replyToId", "channelName".
    user_id_map is a dict mapping "character" -> user_id in the DB.
    channel_id_map is a dict mapping channelName -> channel_id in the DB.
    """
    msg_id = message_data["id"]
    author_name = message_data["character"]
    content = message_data["content"]
    date_str = message_data["date"]
    channel_name = message_data["channelName"]
    reply_to_id = message_data.get("replyToId")  # will be None if not present

    # Convert date_str to a datetime
    created_at = datetime.datetime.fromisoformat(date_str)

    # Map from character name -> user_id
    author_id = user_id_map.get(author_name)
    if not author_id:
        raise ValueError(f"No user_id found for character {author_name}")

    # Map from channelName -> channel_id
    channel_id = channel_id_map.get(channel_name)
    if not channel_id:
        raise ValueError(f"No channel_id found for channel {channel_name}")

    with conn.cursor() as cur:
        # Only include replyToId if it exists and references a valid message
        if reply_to_id:
            cur.execute('SELECT id FROM "Message" WHERE id = %s', (reply_to_id,))
            if not cur.fetchone():
                reply_to_id = None

        # Insert the message
        cur.execute("""
            INSERT INTO "Message" (id, content, "channelId", "authorId", "createdAt", "updatedAt", "replyToId")
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            msg_id,
            content,
            channel_id,
            author_id,
            created_at,
            created_at,
            reply_to_id
        ))
    # We do not commit yet; we'll do that outside in a bulk commit approach.

def main():
    # 1) Generate the messages
    print("Generating 80 messages via GPT...")
    all_messages = generate_messages_via_openai()

    # 2) Connect to DB
    print("Connecting to Postgres...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"[ERROR] Could not connect to DB: {e}")
        sys.exit(1)

    # 3) Ensure each character has a user in the DB, build a map: character -> user_id
    print("Ensuring each of the 4 characters has a user record...")
    user_id_map = {}
    with conn:
        for character_name, character_data in CHARACTERS.items():
            user_id = get_or_create_user(conn, character_name, character_data)
            user_id_map[character_name] = user_id

    # 4) Make sure any channelName references in the JSON exist, build map channelName -> channel_id
    #    We’ll scan all_messages for channelName usage
    channel_names = set([m["channelName"] for m in all_messages])
    channel_id_map = {}
    # We need a "creator_id" for each channel. Let's just pick crumbqueen by default:
    default_creator_id = user_id_map["crumbqueen"]

    with conn:
        for ch_name in channel_names:
            channel_id_map[ch_name] = get_or_create_channel(conn, ch_name, default_creator_id)

    # 5) Insert each message
    print("Inserting messages into DB (this may take a while for 80 messages)...")
    with conn:
        for msg in all_messages:
            insert_message(conn, msg, user_id_map, channel_id_map)

    # 6) Commit all changes
    conn.commit()
    conn.close()

    print("✅ Seeding complete! 80 messages inserted (assuming no errors).")

if __name__ == "__main__":
    main()

