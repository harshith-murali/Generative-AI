from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=10000
)

# Load JSON Schema
with open("review_schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

structured_model = model.with_structured_output(schema)

# Read all reviews
with open("reviews.txt", "r", encoding="utf-8") as f:
    reviews = f.read().strip().split("\n\n")

# Process each review
for i, review in enumerate(reviews, start=1):
    result = structured_model.invoke(review)

    print(f"\n{'=' * 60}")
    print(f"Review {i}")
    print(f"{'=' * 60}")
    print("Original Review:")
    print(review)

    print("\nStructured Output:")
    print(result)