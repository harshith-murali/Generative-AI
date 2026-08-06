from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

documnets = [
    "This is a sample document for embedding.",
    "Another document to test embeddings.",
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072,
)

embedding_vector = embeddings.embed_documents(documnets)

print(str(embedding_vector))