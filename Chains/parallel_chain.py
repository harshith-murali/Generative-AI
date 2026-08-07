from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# -------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# Models
# -------------------------------------------------------------
model1 = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
)

model2 = ChatAnthropic(
    model="claude-sonnet-5",
)

# -------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------
prompt1 = PromptTemplate(
    template="""
Generate short and simple notes from the following text:

{text}
""",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="""
Generate a quiz consisting of 5 multiple-choice questions (MCQs)
based on the following text:

{text}
""",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""
Merge the following notes and quiz into a well-structured study guide.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

# -------------------------------------------------------------
# Output Parser
# -------------------------------------------------------------
parser = StrOutputParser()

# -------------------------------------------------------------
# Parallel Chain
# -------------------------------------------------------------
parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quiz=prompt2 | model2 | parser
)

# -------------------------------------------------------------
# Merge Chain
# -------------------------------------------------------------
merge_chain = prompt3 | model2 | parser

# -------------------------------------------------------------
# Complete Chain
# -------------------------------------------------------------
chain = parallel_chain | merge_chain

# -------------------------------------------------------------
# Sample Input (~200 words)
# -------------------------------------------------------------
text = """
Artificial Intelligence (AI) is a branch of computer science focused on
building systems capable of performing tasks that typically require human
intelligence. These tasks include learning from experience, recognizing
patterns, understanding natural language, solving problems, and making
decisions. Modern AI is powered by machine learning and deep learning,
where algorithms improve their performance by analyzing large amounts of
data.

AI is widely used in everyday life. Virtual assistants like Siri and Alexa
understand voice commands, recommendation systems on Netflix and YouTube
suggest personalized content, and navigation apps determine the fastest
routes using real-time traffic data. In healthcare, AI helps doctors detect
diseases from medical images and predict patient outcomes. In finance, it
is used for fraud detection and risk assessment.

Despite its benefits, AI also presents challenges such as bias in training
data, privacy concerns, job displacement due to automation, and ethical
questions about decision-making. Researchers and policymakers are working
to ensure that AI systems are transparent, fair, and safe for society.

As AI technology continues to evolve, it is expected to transform industries
such as education, transportation, manufacturing, and scientific research,
making many processes faster, more efficient, and more intelligent.
"""

# -------------------------------------------------------------
# Execute Chain
# -------------------------------------------------------------
result = chain.invoke({
    "text": text
})

print(result)
chain.get_graph().print_ascii()