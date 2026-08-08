from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Now, explain this joke in a simple way:\n{joke}",
    input_variables=["joke"],
)

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "programming"}))