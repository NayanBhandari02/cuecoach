import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="CueCoach", page_icon="🎱", layout="centered")

st.title("CueCoach")
st.write("Ask questions from your billiards documents.")

question = st.text_area(
    "Your question",
    placeholder="Example: What is a legal opening break shot in pyramid?",
    height=120,
)

mode = st.radio(
    "Answer style",
    options=["strict", "explain"],
    horizontal=True,
)

with st.expander("Advanced options"):
    top_k = st.slider("Top K", min_value=2, max_value=12, value=5, step=1)
    min_score = st.number_input("Minimum score", min_value=0.0, max_value=1.0, value=0.42, step=0.01)
    max_context_chars = st.number_input("Max context chars", min_value=1000, max_value=50000, value=12000, step=1000)

if st.button("Ask", use_container_width=True):
    question_clean = question.strip()

    if not question_clean:
        st.warning("Enter a question.")
    else:
        payload = {
            "question": question_clean,
            "mode": mode,
            "top_k": top_k,
            "min_score": min_score,
            "max_context_chars": int(max_context_chars),
        }

        try:
            with st.spinner("Thinking..."):
                response = requests.post(API_URL, json=payload, timeout=120)

            if response.status_code != 200:
                st.error(f"API error: {response.status_code}")
                st.code(response.text)
            else:
                data = response.json()
                st.subheader("Answer")
                st.write(data.get("answer", ""))

                st.caption(f"Mode: {data.get('mode', mode)}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure FastAPI is running on http://127.0.0.1:8000.")
        except requests.exceptions.Timeout:
            st.error("The request timed out.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")