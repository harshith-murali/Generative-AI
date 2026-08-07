from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# -------------------------------------------------------------
# Load environment variables (.env)
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# Create the Claude Haiku LLM
# -------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.2
)

# -------------------------------------------------------------
# Define the Pydantic Model
# This specifies the exact structure that we expect from the LLM.
# -------------------------------------------------------------
class TravelDestination(BaseModel):
    destination: str = Field(description="Name of the travel destination")
    country: str = Field(description="Country where the destination is located")
    best_time_to_visit: str = Field(description="Best season or months to visit")
    famous_for: List[str] = Field(description="Popular attractions or things the destination is known for")
    average_budget_per_day: int = Field(description="Estimated daily budget in USD")
    must_try_food: str = Field(description="A local dish every traveler should try")

# -------------------------------------------------------------
# Create the Pydantic Output Parser
# -------------------------------------------------------------
parser = PydanticOutputParser(pydantic_object=TravelDestination)

# -------------------------------------------------------------
# Prompt Template
# The parser automatically generates formatting instructions
# that tell Claude how to structure the response.
# -------------------------------------------------------------
template = PromptTemplate(
    template="""
Generate details for one travel destination.

{format_instructions}
""",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# -------------------------------------------------------------
# Create the LCEL Chain
#
# PromptTemplate
#        │
#        ▼
# Claude Haiku
#        │
#        ▼
# PydanticOutputParser
#        │
#        ▼
# TravelDestination Object
# -------------------------------------------------------------
chain = template | llm | parser

# -------------------------------------------------------------
# Execute the chain
# -------------------------------------------------------------
result = chain.invoke({})

# -------------------------------------------------------------
# Display the output
# -------------------------------------------------------------
print(result)

print("\nDestination:", result.destination)
print("Country:", result.country)
print("Best Time:", result.best_time_to_visit)
print("Famous For:", result.famous_for)
print("Budget per Day (USD):", result.average_budget_per_day)
print("Must Try Food:", result.must_try_food)