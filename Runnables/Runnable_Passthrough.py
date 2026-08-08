from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence

load_dotenv()

passthrough = RunnablePassthrough()

prompt1 = PromptTemplate(
    template="Write a short story about {topic}.",
    input_variables=["topic"],
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Now, summarize this story in one sentence:\n{story}",
    input_variables=["story"],
)

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

# First generate story, then map it into a dict for parallel processing
parallel_chain = RunnableParallel({
    'story': passthrough,
    'summary': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, lambda story: {"story": story}, parallel_chain)

print(final_chain.invoke({"topic": "a robot learning to love"}))