from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------
# Load environment variables (.env)
# This loads the Hugging Face API key and any other environment
# variables required by the application.
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# Create the base LLM using Hugging Face Inference API.
#
# repo_id : The model to be used.
# task    : Specifies that we want text generation.
# -------------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

# -------------------------------------------------------------
# Wrap the endpoint inside ChatHuggingFace.
#
# ChatHuggingFace provides a chat-model interface that works
# seamlessly with LangChain's prompt templates and chains.
# -------------------------------------------------------------
model = ChatHuggingFace(llm=llm)

# =============================================================
# Prompt Template 1
# Generates a detailed report on a given topic.
# =============================================================
template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

# =============================================================
# Prompt Template 2
# Takes the generated report and summarizes it into
# approximately 5–10 sentences.
# =============================================================
template2 = PromptTemplate(
    template="Summarize the following report in 5-10 sentences: {report}",
    input_variables=["report"],
)

# =============================================================
# Traditional (Manual) Workflow
# =============================================================

# Step 1: Fill the first prompt with the topic.
prompt1 = template1.invoke({
    "topic": "Black holes and their impact on the universe"
})

# Step 2: Send the prompt to the LLM to generate the report.
result = model.invoke(prompt1)

# Step 3: Insert the generated report into the second prompt.
prompt2 = template2.invoke({
    "report": result.content
})

# Step 4: Ask the LLM to summarize the report.
summary = model.invoke(prompt2)

# Uncomment to see intermediate outputs.
# print("Detailed Report:\n", result.content)
# print("\nSummary:\n", summary.content)

# =============================================================
# Output Parser
# =============================================================

# StrOutputParser converts the AIMessage returned by the model
# into a plain Python string.
#
# Without this parser:
#     model -> AIMessage
#
# With this parser:
#     model -> String
#
# This is important because the next PromptTemplate expects
# plain text as its input.
parser = StrOutputParser()

# =============================================================
# LangChain Expression Language (LCEL) Chain
# =============================================================
#
# Flow:
#
# User Topic
#      │
#      ▼
# Prompt Template 1
#      │
#      ▼
# Chat Model
#      │
#      ▼
# StrOutputParser
#      │
#      ▼
# Prompt Template 2
#      │
#      ▼
# Chat Model
#      │
#      ▼
# StrOutputParser
#      │
#      ▼
# Final Summary
#
# The output from each component automatically becomes the input
# to the next component.
# =============================================================

chain = (
    template1
    | model
    | parser
    | template2
    | model
    | parser
)

# Execute the complete pipeline.
final_result = chain.invoke({
    "topic": "The history of Dinosaurs and their extinction"
})

# Display the final summarized result.
print("\nFinal Result:\n")
print(final_result)