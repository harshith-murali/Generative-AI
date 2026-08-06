from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072,
)

embedding_vector = embeddings.embed_query("Hello, world!")

print(str(embedding_vector))