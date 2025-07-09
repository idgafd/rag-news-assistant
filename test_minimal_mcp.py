# from app2 import chat_with_mcp_rag
from qdrant_client_wrapper import get_latest_date_int

if __name__ == "__main__":
    # query = "What's up with AI in April 2024?"
    # answer, sources, tool_calls = chat_with_mcp_rag(query)
    #
    # print(f"Answer: {answer}")
    # print(f"Sources: {sources}")
    # print(f"Tool calls: {len(tool_calls)}")
    #
    # for i, call in enumerate(tool_calls):
    #     print(f"\nCall {i + 1}: {call['name']}")
    #     print(f"Arguments: {call['arguments']}")
    #     print(f"Result keys: {list(call['result'].keys()) if isinstance(call['result'], dict) else 'Not dict'}")

    get_latest_date_int()

