from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary of the following poem:\n\n{poem}",
    input_variables=["poem"],
)

chain = prompt | model | parser

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

print(chain.invoke({"poem": docs[0].page_content}))