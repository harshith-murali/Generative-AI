from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chat = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000
)

prompt = """Whats 2*2+5-9+100?"""

response = chat.invoke(prompt)
print(response.text)