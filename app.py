import streamlit as st
from openai import OpenAI

from openai_wrapper import route_user_query, answer_user_query, fallback_answer_user_query
from core_pipeline import execute_routed_call, format_resources_block
from config import OPENAI_API_KEY


api_key = OPENAI_API_KEY
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the .env file")
client = OpenAI(api_key=api_key)


st.set_page_config(page_title="AI News Assistant", layout="wide")
st.title("🧠 AI News Assistant")
st.write("Ask a question and optionally upload an image. The assistant will retrieve relevant articles and generate an answer.")

user_query = st.text_input("Your question")
uploaded_image = st.file_uploader("Optional image", type=["jpg", "jpeg", "png"])

if st.button("Submit") and user_query:
    with st.spinner("Thinking..."):

        image_path = None
        if uploaded_image:
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        call = route_user_query(user_query, image_path=image_path)
        articles = execute_routed_call(call)

        if articles:
            answer = answer_user_query(user_query, articles)
            resources = format_resources_block(articles)
            final_response = f"{answer}\n\n{resources}"
        else:
            final_response = fallback_answer_user_query(user_query)

        st.markdown(final_response)
