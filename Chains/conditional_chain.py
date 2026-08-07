from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.runnables import RunnableLambda, RunnableBranch

# -------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# Create Claude Haiku Model
# -------------------------------------------------------------
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
)

# -------------------------------------------------------------
# Define Output Schema
# -------------------------------------------------------------
class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the customer feedback."
    )

# -------------------------------------------------------------
# Output Parsers
# -------------------------------------------------------------
sentiment_parser = PydanticOutputParser(
    pydantic_object=FeedbackSentiment
)

text_parser = StrOutputParser()

# -------------------------------------------------------------
# Sentiment Classification Prompt
# -------------------------------------------------------------
classification_prompt = PromptTemplate(
    template="""
Classify the sentiment of the following customer feedback.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": sentiment_parser.get_format_instructions()
    },
)

# -------------------------------------------------------------
# Sentiment Classification Chain
# -------------------------------------------------------------
classifier_chain = (
    classification_prompt
    | model
    | sentiment_parser
)

# -------------------------------------------------------------
# RunnableLambda
#
# Converts the Pydantic object into a dictionary so that
# RunnableBranch can easily evaluate conditions.
# -------------------------------------------------------------
prepare_input = RunnableLambda(
    lambda sentiment: {
        "feedback": feedback,
        "sentiment": sentiment.sentiment
    }
)

# -------------------------------------------------------------
# Response Prompts
# -------------------------------------------------------------
positive_prompt = PromptTemplate(
    template="""
The customer wrote:

{feedback}

Write a warm and friendly thank-you response.
""",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="""
The customer wrote:

{feedback}

Write a professional apology and assure the customer
that improvements will be made.
""",
    input_variables=["feedback"],
)

neutral_prompt = PromptTemplate(
    template="""
The customer wrote:

{feedback}

Write a polite acknowledgement.
""",
    input_variables=["feedback"],
)

# -------------------------------------------------------------
# RunnableBranch
# -------------------------------------------------------------
branch = RunnableBranch(

    (
        lambda x: x["sentiment"] == "positive",
        positive_prompt | model | text_parser,
    ),

    (
        lambda x: x["sentiment"] == "negative",
        negative_prompt | model | text_parser,
    ),

    (
        lambda x: x["sentiment"] == "neutral",
        neutral_prompt | model | text_parser,
    ),

    neutral_prompt | model | text_parser,
)

# -------------------------------------------------------------
# Complete Chain
# -------------------------------------------------------------
chain = (
    classifier_chain
    | prepare_input
    | branch
)

# -------------------------------------------------------------
# User Feedback
# -------------------------------------------------------------
feedback = (
    "I absolutely love this product! "
    "It has exceeded all my expectations."
)

# -------------------------------------------------------------
# Execute Chain
# -------------------------------------------------------------
result = chain.invoke({
    "feedback": feedback
})

print(result)