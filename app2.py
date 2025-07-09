import streamlit as st
from openai import OpenAI
import json
import random
from typing import Dict, List, Any
from config import OPENAI_API_KEY

# OpenAI Function Calling Tools - defines available tools for AI
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_articles",
            "description": "Search for articles using text similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The search query/prompt to find relevant articles"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of articles to return",
                        "default": 5
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional start date filter in YYYY-MM-DD format"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional end date filter in YYYY-MM-DD format"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_articles_by_date_range",
            "description": "Get articles from a specific date range",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    }
                },
                "required": ["date_from", "date_to"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_by_uploaded_image",
            "description": "Find articles similar to an uploaded image",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path_or_url": {
                        "type": "string",
                        "description": "Path to local image file or URL to image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional text prompt to help with image search"
                    }
                },
                "required": ["image_path_or_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_comprehensive_answer",
            "description": "Generate a comprehensive answer based on found articles",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The original user question"
                    },
                    "articles": {
                        "type": "array",
                        "description": "List of article dictionaries from search functions",
                        "items": {"type": "object"}
                    },
                    "model": {
                        "type": "string",
                        "description": "OpenAI model to use",
                        "default": "gpt-4-turbo"
                    }
                },
                "required": ["user_query", "articles"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_fallback_answer",
            "description": "Generate fallback answer when no articles are found",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user's question"
                    }
                },
                "required": ["user_query"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Execute a tool (function) with given arguments.
    This replaces your old execute_routed_call.
    """
    try:
        if tool_name == "search_similar_articles":
            from utils import search_similar_articles
            articles = search_similar_articles(
                prompt=arguments["prompt"],
                top_k=arguments.get("top_k", 5),
                date_from=arguments.get("date_from"),
                date_to=arguments.get("date_to")
            )
            # Convert to dictionary format
            articles_dict = []
            for article in articles:
                articles_dict.append({
                    'payload': article.payload,
                    'score': getattr(article, 'score', 0.0),
                    'id': getattr(article, 'id', None)
                })
            return {
                "articles": articles_dict,
                "total_found": len(articles),
                "function_used": "search_similar_articles"
            }

        elif tool_name == "get_articles_by_date_range":
            from utils import get_articles_by_date_range
            articles = get_articles_by_date_range(
                date_from=arguments["date_from"],
                date_to=arguments["date_to"]
            )
            articles_dict = []
            for article in articles:
                articles_dict.append({
                    'payload': article.payload,
                    'score': getattr(article, 'score', 0.0),
                    'id': getattr(article, 'id', None)
                })
            return {
                "articles": articles_dict,
                "total_found": len(articles),
                "function_used": "get_articles_by_date_range"
            }

        elif tool_name == "find_by_uploaded_image":
            from utils import find_by_uploaded_image
            articles = find_by_uploaded_image(
                image_path_or_url=arguments["image_path_or_url"],
                prompt=arguments.get("prompt")
            )
            articles_dict = []
            for article in articles:
                articles_dict.append({
                    'payload': article.payload,
                    'score': getattr(article, 'score', 0.0),
                    'id': getattr(article, 'id', None)
                })
            return {
                "articles": articles_dict,
                "total_found": len(articles),
                "function_used": "find_by_uploaded_image"
            }

        elif tool_name == "generate_comprehensive_answer":
            from openai_client_wrapper import answer_user_query
            from core_pipeline import format_resources_block

            articles = arguments["articles"]
            user_query = arguments["user_query"]
            model = arguments.get("model", "gpt-4-turbo")

            if not articles:
                return {
                    "answer": "No articles provided for answer generation.",
                    "sources_formatted": "No sources available."
                }

            # Convert back to point format for your existing functions
            class ArticlePoint:
                def __init__(self, article_dict):
                    self.payload = article_dict['payload']
                    self.score = article_dict.get('score', 0.0)
                    self.id = article_dict.get('id', None)

            article_points = [ArticlePoint(article) for article in articles]

            # Use your existing functions
            answer = answer_user_query(user_query, article_points, model=model)
            sources = format_resources_block(article_points)

            return {
                "answer": answer,
                "sources_formatted": sources,
                "articles_processed": len(articles)
            }

        elif tool_name == "generate_fallback_answer":
            from openai_client_wrapper import fallback_answer_user_query
            return {
                "answer": fallback_answer_user_query(arguments["user_query"]),
                "sources_formatted": "",
                "fallback_used": True
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"error": f"Error executing {tool_name}: {str(e)}"}


def intelligent_rag_chat(user_query: str, image_path: str = None) -> tuple[str, str, List[Dict]]:
    """
    Intelligent RAG chat using OpenAI function calling.
    Replaces your old route_user_query -> execute_routed_call -> answer_user_query flow.

    Returns: (answer, sources, execution_log)
    """
    api_key = OPENAI_API_KEY
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": """You are an intelligent AI assistant with access to a comprehensive article search system.

Available tools:
- search_similar_articles: Find articles using semantic similarity
- get_articles_by_date_range: Get articles from specific date periods
- find_by_uploaded_image: Find articles similar to uploaded images
- generate_comprehensive_answer: Create detailed answers from articles
- generate_fallback_answer: Provide general answers when no articles found

Strategy:
1. Analyze the user's query to choose the best search method
2. Search for relevant articles
3. If articles are found, generate a comprehensive answer with sources
4. If no articles found, provide a helpful fallback response

Always prioritize providing accurate, well-sourced information."""
        },
        {
            "role": "user",
            "content": user_query + (f"\n[Image uploaded: {image_path}]" if image_path else "")
        }
    ]

    execution_log = []
    final_answer = ""
    sources_formatted = ""
    collected_articles = []
    max_iterations = 10

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
            temperature=0.7
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # If no tool calls, we have the final answer
        if not response_message.tool_calls:
            final_answer = response_message.content
            break

        # Execute all tool calls
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Smart logic for generate_comprehensive_answer
            if function_name == "generate_comprehensive_answer":
                if "articles" not in function_args or not function_args["articles"]:
                    if collected_articles:
                        # Automatically add collected articles
                        function_args["articles"] = collected_articles
                    else:
                        # Switch to fallback if no articles available
                        function_name = "generate_fallback_answer"
                        function_args = {"user_query": user_query}

            # Execute the tool
            tool_result = execute_tool(function_name, function_args)

            # Collect articles from search functions
            if function_name in ["search_similar_articles", "get_articles_by_date_range", "find_by_uploaded_image"]:
                if isinstance(tool_result, dict) and "articles" in tool_result:
                    new_articles = tool_result["articles"]
                    collected_articles.extend(new_articles)

            # Log execution for debugging
            execution_log.append({
                "iteration": iteration + 1,
                "tool": function_name,
                "arguments": function_args,
                "result_summary": {
                    "success": "error" not in tool_result,
                    "articles_found": tool_result.get("total_found", 0) if isinstance(tool_result, dict) else 0,
                    "type": type(tool_result).__name__
                }
            })

            # Extract sources if this is answer generation
            if function_name == "generate_comprehensive_answer" and isinstance(tool_result, dict):
                if "sources_formatted" in tool_result:
                    sources_formatted = tool_result["sources_formatted"]
                if "answer" in tool_result:
                    # Sometimes AI might not make a final response, so we take it from tool result
                    if not final_answer:
                        final_answer = tool_result["answer"]

            elif function_name == "generate_fallback_answer" and isinstance(tool_result, dict):
                if "answer" in tool_result:
                    if not final_answer:
                        final_answer = tool_result["answer"]
                sources_formatted = ""

            # Add result back to conversation
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(tool_result, default=str)
            })

    return final_answer, sources_formatted, execution_log


# Streamlit application
st.set_page_config(page_title="AI News Assistant", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 AI News Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by OpenAI Function Calling - intelligently orchestrated research.<br>Ask anything. Upload images. Get comprehensive answers with sources.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

col_query, col_image = st.columns([2, 1])

with col_query:
    st.markdown("##### What's on your mind today?")
    user_query = st.text_input("", placeholder="e.g. What are the latest developments in AI?")

with col_image:
    st.markdown("##### Optional image context")
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("---")
submit_button = st.button("🚀 Research & Answer")

if submit_button and user_query:
    with st.spinner("AI is intelligently researching your question..."):

        image_path = None
        if uploaded_image:
            image_path = f"/tmp/{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        # Use intelligent RAG chat
        answer, sources, execution_log = intelligent_rag_chat(user_query, image_path=image_path)

        st.markdown("---")

        # Header variants (your originals)
        fun_headers = [
            "Your answer has arrived",
            "Assembled from the digital archives",
            "Processed by silicon neurons",
            "Generated with artificial confidence",
            "Synthesized from the data streams",
            "Computed with digital precision",
            "Extracted from the knowledge base",
            "Delivered by algorithmic intelligence"
        ]
        st.subheader(random.choice(fun_headers))

        # Show the answer
        if answer:
            st.write(answer)
        else:
            st.warning("Couldn't generate an answer. Check the execution log below for details.")

        # Show how AI worked
        if execution_log:
            st.markdown("### 🔍 AI Execution Process")
            with st.expander("See how AI solved your query"):
                for step in execution_log:
                    status = "✅" if step["result_summary"]["success"] else "❌"
                    st.markdown(f"**Step {step['iteration']}: {step['tool']}** {status}")

                    if step["result_summary"]["articles_found"] > 0:
                        st.info(f"Found {step['result_summary']['articles_found']} articles")

                    # Show arguments in a code block instead of nested expander
                    st.markdown(f"**Arguments for {step['tool']}:**")
                    st.code(json.dumps(step['arguments'], indent=2), language='json')

        # Show sources
        st.markdown("### 📚 Sources")
        if sources and sources.strip():
            with st.expander("View sources used in research"):
                st.markdown(sources)
        else:
            st.info("No specific sources found for this query.")

# Sidebar with information
with st.sidebar:
    st.markdown("### 🧠 How It Works")
    st.markdown("""
    This assistant uses **OpenAI Function Calling** to intelligently:

    1. **Analyze** your question
    2. **Choose** the best search strategy
    3. **Find** relevant articles
    4. **Generate** comprehensive answers
    5. **Provide** sources automatically
    """)

    st.markdown("### 🛠️ Available Tools")
    st.markdown("""
    - **🔍 Semantic Search**: Find articles by meaning
    - **📅 Date Search**: Get articles from specific periods
    - **🖼️ Image Search**: Find articles similar to images
    - **📝 Answer Generation**: Create comprehensive responses
    - **🔄 Fallback**: Handle edge cases gracefully
    """)

    if st.button("🧪 Test System"):
        with st.spinner("Testing all tools..."):
            try:
                test_result = execute_tool("generate_fallback_answer", {"user_query": "test"})
                if "error" not in test_result:
                    st.success("✅ All systems operational!")
                else:
                    st.error(f"❌ System error: {test_result['error']}")
            except Exception as e:
                st.error(f"❌ Test failed: {e}")

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Be specific** for better results
    - **Include dates** for time-sensitive queries
    - **Upload images** for visual context
    - **Ask follow-ups** for deeper insights
    """)