import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import json

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

client = QdrantClient(
    url="http://localhost:6333"
)

with open("knowledge.json", "r", encoding="utf-8") as file:
    data = json.load(file)


COLLECTION_NAME = "advanced-rag"

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

points=[]

for idx, item in enumerate(data):
    text=item["text"]
    embedding=embedding_model.encode(text)
    points.append(PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "text": text,
            "is_active": item["is_active"],
            "category": item["category"]
        }
    ))

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="is_active",
    field_schema=PayloadSchemaType.BOOL
)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

def search(query, category=None, is_active=None):

    query_embedding = embedding_model.encode(query).tolist()

    must_conditions = []

    if category:
        must_conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category)
            )
        )

    if is_active is not None:
        must_conditions.append(
            FieldCondition(
                key="is_active",
                match=MatchValue(value=is_active)
            )
        )

    query_filter = Filter(
        must=must_conditions
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=2,
        with_payload=True
    ).points

    return results


query="What technologies are used to build applications?"
category=None
is_active=None

results = search(query, category=category, is_active=is_active)

for result in results:
    print("Query:", query)
    print("Score:", result.score)
    print("Text:", result.payload["text"])
    print("Category:", result.payload["category"])
    print("Active:", result.payload["is_active"])
    print("-" * 50)

print("=" * 50)
print("Retrieved Context:")
print("=" * 50)

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

                {query}
            """
        }
    ],

    temperature=0
)

print("Answer:")
print(response.choices[0].message.content)