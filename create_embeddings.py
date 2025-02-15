import json
import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv 

load_dotenv()
# ✅ Load API Key securely
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ OpenAI API Key is missing! Set it as an environment variable.")

openai.api_key = API_KEY

# ✅ Load the Bhagavad Gita data
with open("processed_gita.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

# ✅ Generate embeddings using OpenAI's "text-embedding-ada-002"
texts = [verse["meaning_in_english"] for verse in verses]  # Extract "Meaning" for embeddings
embeddings = []

for text in texts:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings.append(response["data"][0]["embedding"])

# ✅ Convert embeddings to NumPy array
embeddings_array = np.array(embeddings).astype("float32")

# ✅ Initialize FAISS index
dimension = embeddings_array.shape[1]  # Get embedding size
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# ✅ Save FAISS index
faiss.write_index(index, "gita_embeddings.index")

print("✅ Embeddings Created & Stored in FAISS Successfully!")


