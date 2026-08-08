from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableSequence , RunnableLambda

load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)

prompt1 = PromptTemplate(
    template = "Write a detailed report about {topic}.",
    input_variables = ["topic"],
)

prompt2 = PromptTemplate(
    template = "Now, summarize this report in one sentence:\n{report}",
    input_variables = ["report"],
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda report: len(report.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic": "Modi in India and his success in running the country"}))