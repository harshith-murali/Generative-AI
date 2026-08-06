from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant specializing in {domain}."),
        ("human", "{paper_input}"),
    ]
)

prompt = chat_template.invoke(
    {
        "domain": "Computer Science",
        "paper_input": "Explain the concept of neural networks in simple terms."
    }
)

print(prompt.messages[0].content)
print(prompt.messages[1].content)