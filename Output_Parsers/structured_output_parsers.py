from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# -------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# Create Claude Haiku LLM
# -------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.0
)

# -------------------------------------------------------------
# Define the output schema
# -------------------------------------------------------------
name_schema = ResponseSchema(
    name="name",
    description="Name of the fictional person"
)

age_schema = ResponseSchema(
    name="age",
    description="Age of the fictional person"
)

city_schema = ResponseSchema(
    name="city",
    description="City where the fictional person lives"
)

profession_schema = ResponseSchema(
    name="profession",
    description="Profession of the fictional person"
)

# -------------------------------------------------------------
# Create the Structured Output Parser
# -------------------------------------------------------------
parser = StructuredOutputParser.from_response_schemas(
    [
        name_schema,
        age_schema,
        city_schema,
        profession_schema,
    ]
)

# -------------------------------------------------------------
# Prompt Template
# -------------------------------------------------------------
template = PromptTemplate(
    template="""
Generate details of a fictional person.

{format_instructions}
""",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# -------------------------------------------------------------
# Create the LCEL Chain
# -------------------------------------------------------------
chain = template | llm | parser

# -------------------------------------------------------------
# Invoke the chain
# -------------------------------------------------------------
result = chain.invoke({})

print(result)