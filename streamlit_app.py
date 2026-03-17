import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="CueCoach", page_icon="🎱", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("CueCoach")
    mode = st.radio(
        "Answer style",
        options=["strict", "explain"],
        index=1,
        horizontal=False,
    )
    top_k = st.slider("Top K", min_value=2, max_value=12, value=5, step=1)
    min_score = st.number_input("Minimum score", min_value=0.0, max_value=1.0, value=0.42, step=0.01)
    max_context_chars = st.number_input("Max context chars", min_value=1000, max_value=50000, value=12000, step=1000)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("CueCoach Chat")
st.caption("Use strict mode for rules. Explain mode for explanation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question about your billiards documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    chat_history = st.session_state.messages[:-1]

    payload = {
        "question": user_input,
        "mode": mode,
        "top_k": top_k,
        "min_score": min_score,
        "max_context_chars": int(max_context_chars),
        "chat_history": chat_history,
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(API_URL, json=payload, timeout=120)

            if response.status_code != 200:
                assistant_text = f"API error: {response.status_code}\n{response.text}"
                st.error(assistant_text)
            else:
                data = response.json()
                assistant_text = data.get("answer", "")
                st.write(assistant_text)
                st.caption(f"Mode: {data.get('mode', mode)}")

    except requests.exceptions.ConnectionError:
        assistant_text = "Could not connect to the API. Make sure FastAPI is running on http://127.0.0.1:8000."
        with st.chat_message("assistant"):
            st.error(assistant_text)
    except requests.exceptions.Timeout:
        assistant_text = "The request timed out."
        with st.chat_message("assistant"):
            st.error(assistant_text)
    except Exception as e:
        assistant_text = f"Unexpected error: {e}"
        with st.chat_message("assistant"):
            st.error(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})