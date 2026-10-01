import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

client = QdrantClient(
    url="http://localhost:6333"
)

with open("documents.txt", "r", encoding="utf-8") as file:
    documents = [
        line.strip()
        for line in file
        if line.strip()
    ]



embeddings = []

for document in documents:
    embedding = embedding_model.encode(document)
    embeddings.append(embedding.tolist())


COLLECTION_NAME = "test_documents"

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


points = []

for i in range(len(documents)):

    point = PointStruct(
        id=i+1,
        vector=embeddings[i],
        payload={
            "text": documents[i]
        }
    )

    points.append(point)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

question = "Which technology is used for caching?"

query_embedding = embedding_model.encode(question).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=2,
    with_payload=True
).points


retrieved_context = "\n\n".join(
    result.payload["text"]
    for result in results
)


response = groq_client.chat.completions.create(

    model="qwen/qwen3.8-27b",

    messages=[
        {
            "role": "system",
            "content": """
                You are a helpful assistant.
                Answer the question using only the provided context.
                If the answer is not present in the context,
                say that you don't know.
            """
        },
        {
            "role": "user",
            "content": f"""
                Context:

                {retrieved_context}

                Question:

                {question}
            """
        }
    ],

    temperature=0
)

print(response.choices[0].message.content)