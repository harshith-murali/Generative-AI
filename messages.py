from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    max_new_tokens=10000,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is langchain ?"),
    # AIMessage(content="I'm here to help you with any questions or tasks you have. What do you need assistance with?"),
]

model_response = model.invoke(messages)

messages.append(AIMessage(content=model_response.content))

print("\nConversation History")
print("-" * 50)

for message in messages:
    if isinstance(message, SystemMessage):
        print(f"🟡 System : {message.content}\n")
    elif isinstance(message, HumanMessage):
        print(f"🟢 You    : {message.content}\n")
    elif isinstance(message, AIMessage):
        print(f"🔵 Bot    : {message.content}\n")