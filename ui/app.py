import streamlit as st
import requests

st.set_page_config(
    page_title="Mini ChatGPT",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Transformer Mini ChatGPT")

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_area("Enter your prompt")

temperature = st.slider("Temperature", 0.1, 2.0, 0.8)

max_tokens = st.slider("Max Tokens", 50, 500, 200)

if st.button("Generate"):

    if prompt.strip() == "":
        st.warning("Enter some text")
    else:

        with st.spinner("Generating..."):

            res = requests.post(
                "http://127.0.0.1:8000/generate",
                json={
                    "text": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=120
            )

            output = res.json()["response"]

            st.session_state.history.append((prompt, output))


st.divider()

for q, a in reversed(st.session_state.history):

    st.markdown("### 🧑 Prompt")
    st.write(q)

    st.markdown("### 🤖 Response")
    st.write(a)