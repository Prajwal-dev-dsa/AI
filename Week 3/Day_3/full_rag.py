import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

groq_model="qwen/qwen3.8-27b"

client=Groq(api_key=my_api_key)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

documents = [
    "Python is a popular programming language used for web development, automation, data science, and artificial intelligence.",

    "React is a JavaScript library for building user interfaces. It uses reusable components and is commonly used for creating modern web applications.",

    "MongoDB is a NoSQL database that stores data in flexible JSON-like documents. It is commonly used with Node.js and Express.",

    "Docker is a platform that packages applications and their dependencies into containers, making applications easier to develop, deploy, and run consistently.",

    "Machine learning is a branch of artificial intelligence where computers learn patterns from data and use those patterns to make predictions or decisions.",

    "Git is a distributed version control system that allows developers to track code changes, create branches, and collaborate with other developers.",

    "FastAPI is a modern Python web framework used for building high-performance APIs. It supports automatic API documentation and type validation.",

    "Redis is an in-memory data store commonly used for caching, session management, rate limiting, and fast temporary data storage."
]

document_embedding=embedding_model.encode(documents)

def match_context(user_prompt):
    user_embedding=embedding_model.encode(user_prompt)
    similarities=[]
    for i, doc in enumerate(document_embedding):
        cosine_score=cosine_similarity(user_embedding, doc)
        similarities.append((cosine_score, documents[i]))
    similarities.sort(reverse=True)
    print(similarities)
    return similarities[0][1]

def ask_llm(user_prompt):
    matched_context = match_context(user_prompt)
    sys_prompt=f"""answer in one line. if question matches with the provided context then use the context to answer the user's question. context: {matched_context}"""

    sys_message={
        "role": "system",
        "content": sys_prompt
    }
    user_message={
        "role": "user",
        "content": user_prompt
    }
    response=client.chat.completions.create(
        model=groq_model,
        messages=[sys_message, user_message]
    )
    return response.choices[0].message.content

query="I need a way for several programmers to safely maintain and experiment with different versions of the same project."
print(ask_llm(query))