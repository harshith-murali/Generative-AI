from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=10000
)


class ReviewSummary(BaseModel):
    summary: str = Field(
        description="A concise summary of the review, capturing the main points and overall sentiment."
    )

    key_themes: list[str] = Field(
        description="A list of the main themes or topics discussed in the review."
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment of the review."
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="A list of the review's positive aspects."
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="A list of the review's negative aspects."
    )


structured_model = model.with_structured_output(ReviewSummary)

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

    # Convert to dictionary if needed
    print("\nAs Dictionary:")
    print(result.model_dump())

    # Convert to JSON if needed
    print("\nAs JSON:")
    print(result.model_dump_json(indent=4))