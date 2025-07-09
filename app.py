import streamlit as st
from openai import OpenAI

from clients.openai_client_wrapper import OpenAIRouterClient
from core_pipeline import execute_routed_call, format_resources_block
# from config import OPENAI_API_KEY

import itertools
import time
import random


# api_key = OPENAI_API_KEY
# if not api_key:
#     raise EnvironmentError("OPENAI_API_KEY is not set in secrets or .env")
# client = OpenAI(api_key=api_key)

openai_client = OpenAIRouterClient()

st.set_page_config(page_title="AI News Assistant", layout="wide")

st.markdown("<h1 style='text-align: center;'>AI News Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Ask something. Anything. Preferably not about the meaning of life.<br>Upload an image if you think it helps, we won't judge.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

col_query, col_image = st.columns([2, 1])

with col_query:
    st.markdown("##### What’s on your mind today, human?")
    user_query = st.text_input("", placeholder="e.g. What's up with AI in April 2024?")

with col_image:
    st.markdown("##### Optional visual evidence")
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("---")
submit_button = st.button("Go fetch the wisdom")

if submit_button and user_query:
    with st.spinner("Consulting the archives..."):

        image_path = None
        if uploaded_image:
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        call = openai_client.route_user_query(user_query, image_path=image_path)
        articles = execute_routed_call(call)

        st.markdown("---")
        subheader_variants = [
            "A possibly intelligent answer",
            "May or may not be hallucinated",
            "This came from the thinking box",
            "Generated in a GPU-powered existential crisis",
            "Not bad for a bunch of math",
            "If this is wrong, it’s a feature, not a bug",
            "Made with <3 and linear algebra",
            "Verbal output of cold, hard matrices",
            "Hallucinated responsibly™",
            "Compiled by algorithms with strong opinions",
            "Definitely not copied from Wikipedia (probably)",
            "Not responsible for any life decisions made after reading this"
        ]
        st.subheader(random.choice(subheader_variants))

        if articles:
            answer = openai_client.answer_user_query(user_query, articles)
            st.write(answer)
        else:
            answer = openai_client.fallback_answer_user_query(user_query)
            st.write(f"*{answer}*")

        st.markdown("### Footnotes from the past (aka. Sources)")
        if articles:
            with st.expander("Click to reveal the boring but important stuff"):
                resources = format_resources_block(articles)
                st.markdown(resources)
        else:
            st.info("Hmm, no ancient scrolls found on that one.")
            
