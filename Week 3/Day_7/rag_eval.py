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
            "doc_id": item["id"],
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

def fn():
    # --------------------------------------------------
    # 1. Load golden dataset
    # --------------------------------------------------
    with open("golden_dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # --------------------------------------------------
    # Iterate through every golden test case
    # --------------------------------------------------
    for idx, item in enumerate(data, start=1):

        query = item["query"]
        reference_answer = item["relevant_answer"]
        relevant_context_ids = set(item["relevant_context_ids"])

        # Optional filters
        category = None
        is_active = None

        print("\n")
        print("=" * 70)
        print(f"TEST CASE {idx}")
        print("=" * 70)

        print("Query:", query)

        # --------------------------------------------------
        # 2. Retrieve relevant documents from Qdrant
        # --------------------------------------------------
        results = search(
            query,
            category=category,
            is_active=is_active
        )

        print("\nRetrieved Documents:")
        print("-" * 70)

        retrieved_context = []
        retrieved_ids = set()

        for result in results:

            # Qdrant point ID
            result_id = result.payload["doc_id"]

            retrieved_ids.add(result_id)

            print("ID:", result_id)
            print("Score:", result.score)
            print("Text:", result.payload["text"])
            print("Category:", result.payload["category"])
            print("Active:", result.payload["is_active"])
            print("-" * 50)

            retrieved_context.append(result.payload["text"])

        # --------------------------------------------------
        # 3. Calculate Retrieval Precision & Recall
        # --------------------------------------------------

        true_positives = len(
            retrieved_ids.intersection(relevant_context_ids)
        )

        retrieved_count = len(retrieved_ids)
        relevant_count = len(relevant_context_ids)

        precision = (
            true_positives / retrieved_count
            if retrieved_count > 0
            else 0
        )

        recall = (
            true_positives / relevant_count
            if relevant_count > 0
            else 0
        )

        print("\nRetrieval Evaluation")
        print("-" * 70)
        print("Relevant IDs :", relevant_context_ids)
        print("Retrieved IDs:", retrieved_ids)
        print("Precision    :", round(precision, 3))
        print("Recall       :", round(recall, 3))

        # --------------------------------------------------
        # 4. Build retrieved context
        # --------------------------------------------------
        retrieved_context = "\n\n".join(retrieved_context)

        print("\n")
        print("=" * 70)
        print("RETRIEVED CONTEXT")
        print("=" * 70)
        print(retrieved_context)

        # --------------------------------------------------
        # 5. LLM #1 -> Generate Answer
        # --------------------------------------------------
        response = groq_client.chat.completions.create(

            model="qwen/qwen3.8-27b",

            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful RAG assistant.

                    Answer the question using ONLY the provided context.

                    Do not use your own outside knowledge.

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

        generated_answer = response.choices[0].message.content

        print("\n")
        print("=" * 70)
        print("GENERATED ANSWER")
        print("=" * 70)
        print(generated_answer)

        # --------------------------------------------------
        # 6. LLM #2 -> Evaluate Generated Answer
        # --------------------------------------------------

        evaluation_response = groq_client.chat.completions.create(

            model="qwen/qwen3.8-27b",

            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert RAG evaluator.

                    Evaluate the generated answer using the query,
                    retrieved context, and reference answer.

                    Give a score from 0 to 1 for each metric:

                    1. Faithfulness
                    - Is every claim in the generated answer supported
                        by the retrieved context?

                    2. Correctness
                    - Does the generated answer correctly answer the question
                        according to the reference answer?

                    3. Relevancy
                    - Does the generated answer directly address the question
                        without unnecessary unrelated information?

                    Return ONLY valid JSON in exactly this format:

                    {
                        "faithfulness": 0.0,
                        "correctness": 0.0,
                        "relevancy": 0.0
                    }

                    Do not return markdown.
                    Do not return explanations.
                    """
                                    },
                                    {
                                        "role": "user",
                                        "content": f"""
                    Question:

                    {query}

                    Retrieved Context:

                    {retrieved_context}

                    Reference Answer:

                    {reference_answer}

                    Generated Answer:

                    {generated_answer}
                    """
                }
            ],

            temperature=0
        )

        # --------------------------------------------------
        # 7. Parse evaluator response
        # --------------------------------------------------
        evaluation_text = evaluation_response.choices[0].message.content

        try:
            evaluation = json.loads(evaluation_text)

            faithfulness = float(
                evaluation["faithfulness"]
            )

            correctness = float(
                evaluation["correctness"]
            )

            relevancy = float(
                evaluation["relevancy"]
            )

        except (json.JSONDecodeError, KeyError, ValueError):

            print("\nEvaluator returned invalid JSON:")
            print(evaluation_text)

            faithfulness = 0
            correctness = 0
            relevancy = 0

        # --------------------------------------------------
        # 8. Print final metrics
        # --------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL EVALUATION")
        print("=" * 70)

        print("Precision    :", round(precision, 3))
        print("Recall       :", round(recall, 3))
        print("Faithfulness :", round(faithfulness, 3))
        print("Correctness  :", round(correctness, 3))
        print("Relevancy    :", round(relevancy, 3))

        print("=" * 70)

fn()