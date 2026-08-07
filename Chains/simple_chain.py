from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}.",
    input_variables = ["topic"]
)

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.8
)

parser = StrOutputParser()

chain = prompt | model | parser # LCEL chain: PromptTemplate -> Claude Haiku -> StrOutputParser

result = chain.invoke({
    "topic": "Football"
})


print(result)
chain.get_graph().print_ascii()