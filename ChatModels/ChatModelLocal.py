from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1000, "temperature": 0.9}
)

model = ChatHuggingFace(llm=llm, max_tokens=1000)

result = model.invoke("Write me an essay about Virat Kohli's cricket career.")

print(result.content)