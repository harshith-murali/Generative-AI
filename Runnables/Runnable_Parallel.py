from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}.",
    input_variables = ["topic"],
)

prompt2 = PromptTemplate(
    template = "Generate a LinkedIn post about {topic}.",
    input_variables = ["topic"],
)

model = ChatAnthropic(
    model = "claude-haiku-4-5-20251001",
    max_tokens = 1000,
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"topic": "AI in healthcare"})

print("Tweet:", result['tweet'])
print("LinkedIn:", result['linkedin'])