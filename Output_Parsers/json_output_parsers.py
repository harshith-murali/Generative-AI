from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# -------------------------------------------------------------
# Load environment variables (.env)
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# Create the Claude Haiku model
# -------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.8
)

# -------------------------------------------------------------
# Create a JSON output parser
# -------------------------------------------------------------
parser = JsonOutputParser()

# -------------------------------------------------------------
# Prompt Template
# The parser automatically provides format instructions that tell
# Claude to return the response as valid JSON.
# -------------------------------------------------------------
template = PromptTemplate(
    template="""
Give me the name, age, and city of a fictional person.

{format_instructions}
""",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# -------------------------------------------------------------
# Build the LCEL chain
#
# PromptTemplate
#        │
#        ▼
# Claude Haiku
#        │
#        ▼
# JsonOutputParser
#        │
#        ▼
# Python Dictionary
# -------------------------------------------------------------
chain = template | llm | parser

# -------------------------------------------------------------
# Execute the chain
# -------------------------------------------------------------
result = chain.invoke({})

print(result)