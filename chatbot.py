import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot")
st.caption("Powered by LangChain + Hugging Face")

# ---------------- Load Model ---------------- #

@st.cache_resource
def load_model():

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        task="text-generation",
        max_new_tokens=1000,
        temperature=0.7,
    )

    return ChatHuggingFace(llm=llm)


model = load_model()

# ---------------- Chat History ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a helpful, friendly AI assistant."
        )
    ]

# ---------------- Display Previous Messages ---------------- #

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ---------------- User Input ---------------- #

prompt = st.chat_input("Ask me anything...")

if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Assistant response
    with st.chat_message("assistant"):

        placeholder = st.empty()

        # Loader while waiting for first token
        with st.spinner("Thinking..."):
            stream = model.stream(st.session_state.messages)

        response_parts = []

        def generate():

            for chunk in stream:

                if chunk.content:
                    response_parts.append(chunk.content)
                    yield chunk.content

        # Stream the response
        placeholder.write_stream(generate())

    # Save assistant response
    full_response = "".join(response_parts)

    st.session_state.messages.append(
        AIMessage(content=full_response)
    )