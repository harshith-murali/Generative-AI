import json

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.5,
    max_tokens=10000,
)

# Load template from JSON
with open("template.json", "r") as file:
    templates = json.load(file)

prompt_template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template=templates["research_summary"],
)

chain = prompt_template | model

st.set_page_config(page_title="Research Assistant")

st.title("Research Assistant")

paper_input = st.selectbox(
    "Select a research paper:",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "GPT-3: Language Models are Few-Shot Learners",
        "GPT-2: Language Models are Unsupervised Multitask Learners",
        "LLaMA: Open and Efficient Foundation Language Models",
        "LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
        "Gemini: A Family of Highly Capable Multimodal Models",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
        "Vision Transformer (ViT)",
        "CLIP: Learning Transferable Visual Models From Natural Language Supervision",
        "Segment Anything",
        "Denoising Diffusion Probabilistic Models",
        "Deep Residual Learning for Image Recognition (ResNet)",
        "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)",
        "Playing Atari with Deep Reinforcement Learning",
    ],
)

style_input = st.selectbox(
    "Select a writing style:",
    [
        "Select...",
        "Academic",
        "Informal",
        "Professional",
        "Code Heavy",
        "Math Heavy",
        "Creative",
        "Humorous",
        "Persuasive",
        "Narrative",
        "Expository",
        "Descriptive",
    ],
)

length_input = st.selectbox(
    "Select a length:",
    [
        "Select...",
        "Short",
        "Medium",
        "Long",
    ],
)

if st.button("Generate Summary"):

    if "Select..." in (paper_input, style_input, length_input):
        st.warning("Please select all the options.")
    else:
        try:
            with st.spinner("Generating summary..."):
                response = chain.invoke(
                    {
                        "paper_input": paper_input,
                        "style_input": style_input,
                        "length_input": length_input,
                    }
                )

            st.markdown(response.content)

        except Exception as e:
            st.exception(e)


# =============================================================================
# Why PromptTemplate instead of f-strings?
#
# 1. Separation of Prompt and Code
#    - Keeps prompt independent from business logic.
#
# 2. Reusability
#    - Same template can be reused with different variables.
#
# 3. Better LangChain Integration
#    - PromptTemplate is a Runnable.
#    - Enables: prompt | model | parser
#
# 4. Easier Pipeline Construction
#    User Input -> PromptTemplate -> LLM -> Output Parser
#
# 5. Better Maintainability
#    - Easier to manage large prompts.
#
# 6. Easy Prompt Swapping
#    - Simply choose another PromptTemplate.
#
# 7. Variable Validation
#    - Raises errors for missing template variables.
#
# 8. Easier Testing
#    - Can inspect formatted prompts before sending to the LLM.
#
# 9. Supports Advanced LangChain Features
#    - ChatPromptTemplate
#    - Few-shot prompting
#    - RAG
#    - Agents
#    - Memory
#
# 10. Production Readability
#     - Prompts become reusable components rather than inline strings.
# =============================================================================