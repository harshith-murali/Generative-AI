from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm, max_tokens=1000)

result = model.invoke("Write me an email to my boss about the progress of the project.")

print(result.content)

