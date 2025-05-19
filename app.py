import streamlit as st
from openai import OpenAI

from openai_wrapper import route_user_query, answer_user_query, fallback_answer_user_query
from core_pipeline import execute_routed_call, format_resources_block
from config import OPENAI_API_KEY

import itertools
import time
import random


api_key = OPENAI_API_KEY
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in secrets or .env")
client = OpenAI(api_key=api_key)

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
st.markdown("""
<style>
div.stButton > button {
    background-color: #e49bff !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6em 1.5em !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    transition: background-color 0.3s ease, transform 0.1s ease !important;
    box-shadow: 0 4px 8px rgba(228, 155, 255, 0.4) !important;
}

div.stButton > button:hover {
    background-color: #c35bfa !important;
    box-shadow: 0 6px 12px rgba(195, 91, 250, 0.5) !important;
}

div.stButton > button:active {
    background-color: #a12ed4 !important;
    transform: scale(0.98) !important;
}
</style>
""", unsafe_allow_html=True)


submit_button = st.button("Go fetch the wisdom")

if submit_button and user_query:
    with st.spinner("Consulting the archives..."):

        image_path = None
        if uploaded_image:
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        call = route_user_query(user_query, image_path=image_path)
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
            answer = answer_user_query(user_query, articles)
            st.write(answer)
        else:
            answer = fallback_answer_user_query(user_query)
            st.write(f"*{answer}*")

        # Resources
        st.markdown("### Footnotes from the past (aka. Sources)")
        if articles:
            with st.expander("Click to reveal the boring but important stuff"):
                resources = format_resources_block(articles)
                st.markdown(resources)
        else:
            st.info("Hmm, no ancient scrolls found on that one.")
            
