import streamlit as st
from openai import OpenAI

from openai_wrapper import route_user_query, answer_user_query, fallback_answer_user_query
from core_pipeline import execute_routed_call, format_resources_block
from config import OPENAI_API_KEY


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
# col1, col2 = st.columns([3, 1])
# with col1:
#     user_query = st.text_input("So, what are you curious about?", placeholder="e.g. What happened in AI last month?")
# with col2:
#     uploaded_image = st.file_uploader("Optional visual evidence", type=["jpg", "jpeg", "png"])

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

        # Save image if uploaded
        image_path = None
        if uploaded_image:
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        # Route → Search → Answer
        call = route_user_query(user_query, image_path=image_path)
        articles = execute_routed_call(call)

        # Display Answer
        st.markdown("---")
        st.subheader("A possibly intelligent answer")
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
            
