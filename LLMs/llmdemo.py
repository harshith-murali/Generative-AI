from langchain_anthropic import AnthropicLLM
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = AnthropicLLM(model="claude-haiku-4-5-20251001" , temperature=0.9, max_tokens_to_sample=1000)

response = llm.invoke("What is the capital of USA?")

print(response)

