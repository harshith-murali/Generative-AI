from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a detailed report on {topic}.",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Based on the following report, summarize them in a concise manner: {facts}",
    input_variables = ["facts"]
)

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.8
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser # LCEL chain: PromptTemplate -> Claude Haiku -> StrOutputParser -> PromptTemplate -> Claude Haiku -> StrOutputParser

result = chain.invoke({
    "topic" : "Politics in India",
})

print(result)
chain.get_graph().print_ascii()
