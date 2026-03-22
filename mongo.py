# mongo.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Use 'project' database
db = client['project']

# Example collections
summary_collection = db['summaries']
chat_collection = db['chats']
