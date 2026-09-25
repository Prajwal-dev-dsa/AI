from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

text="Prajwal is a final-year Computer Science student."

embedding=model.encode(text)

print(embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

t1="I love machine learning"
t2="prajwal is a bad boy"

e1=model.encode(t1)
e2=model.encode(t2)

print(f"Similarity between '{t1}' and '{t2}': {cosine_similarity(e1, e2)}")