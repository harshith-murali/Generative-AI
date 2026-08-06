from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is one of the greatest batsmen...",
    "Mahendra Singh Dhoni, popularly known as MS Dhoni...",
    'Chris Gayle is one of the most destructive batsmen and is widely known as the "Universe Boss."...',
    "David Warner is one of Australia's finest opening batsmen..."
]

doc_embeddings = embedding.embed_documents(documents)

query = "Tell me about Bumrah's bowling style and achievements."
query_embedding = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

for i, score in enumerate(similarities):
    print(f"Document {i}: {score:.4f}")

best_match = similarities.argmax()

print("\nMost relevant document:")
print(documents[best_match])