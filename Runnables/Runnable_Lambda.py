from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence , RunnableLambda

load_dotenv()

# Can wrap a function in a RunnableLambda to make it a Runnable or use direct lambda functions. This is useful for simple transformations or computations that don't require a full model call.
# def word_counter(text: str) -> int:
#     return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)

passthrough = RunnablePassthrough()

prompt = PromptTemplate(
    template="Write a short story about {topic}.",
    input_variables=["topic"],
)

parser = StrOutputParser()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)

story_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'story': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x : len(x.split()))
})

final_chain = RunnableSequence(story_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "A flower that can talk"}))