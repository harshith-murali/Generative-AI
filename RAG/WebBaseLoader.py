from langchain_community.document_loaders import WebBaseLoader

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the following question based on the content of the webpage:\n\n{question}\n\nWebpage Content:\n{content}",
    input_variables=["question", "content"],
)
url = 'https://www.amazon.in/Apple-2026-MacBook-Laptop-chip/dp/B0GR1K8S8H/ref=sr_1_13?crid=D7W2SVHM5VLK&dib=eyJ2IjoiMSJ9.uJqKZXr8V5pE53mUkQd4Ipqk3zqeZ9FV0defA51j8lKec_IqSCv9GKMWaDjOUCk0mZ-aMbvZPL1Vsb3-KNDlzen74wmBK3vSmaWvPSfLIA8qK1nJ2QDJ8tuzgmCvyhA_lm8599Rdj_xlStMq-gx_Uu3j0xN3dHad98DOGvWKEpGCRO6B0XTiFX5C3v_bhdIsj5P3bc7UiHWGbfsm8Y4i-7iFDaDX8Ofqvl_f5RVcsQPsjSTEytYBtcyu7P-8sDFCNyf4DVxBAf4yFJLvWnBA9JdZPVOtWLmPGmrP7xXNSZo.XLUKWeCtouaIByXLWMYAmCiVJOOHWynOnoD0MpJhwls&dib_tag=se&keywords=macbook%2Bair&qid=1786204287&s=electronics&sprefix=macbook%2Bai%2Celectronics%2C289&sr=1-13&th=1'

loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser 

result = chain.invoke({
    "question": "What is the price of the Apple 2026 MacBook Air Laptop with M5 chip?",
    "content": docs[0].page_content
})

print(result)