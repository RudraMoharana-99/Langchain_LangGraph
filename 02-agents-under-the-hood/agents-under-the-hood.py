from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable


MAX_ITERATION = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """Look up a product price in catalog.."""
    print(f"    Executing get_product_price(product= {product})")
    prices = {"laptop": 1299.99, "headphones": 149.99, "keyboard": 145.45}

    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
       Available tiers: bronze, silver and gold"""

    print(f"    Executing apply_discount(price: {price}, discount_tier: {discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1-discount/100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}

    # --------Initialize LLM -------------
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools=tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have acess to product catalog tool"
                "and a discount tool \n\n"
                "STRICT RULE - you must follow this exactly:\n"
                "1. Never guess or assume any product price. "
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you recieved "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number. \n"
                "3. Never calculate discount yourself using math,"
                "Always use the apply_discount tool.\n"
                "4. If the user doesn't specify discount tier,"
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question), 
    ]

    for iteration in range(1, MAX_ITERATION+1):
        print(f"\n--- Iteration : {iteration}---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\Final answer: {ai_message.content}")
            return ai_message.content

        # Process only first tool call - force on tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"    [Tool Selcted]  {tool_name} | [With Args] {tool_args}")
        tools_to_use = tool_dict.get(tool_name)

        if tools_to_use is None:
            raise ValueError(f"tool {tool_name} not found")

        observation = tools_to_use.invoke(tool_args)

        print(f"[Tools Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: max iteration reached without a final answer.")
    return

if __name__ == "__main__":
    print("Hello LangCahin Agent (.bind tool)")
    print()
    result = run_agent(question="What is the price of laptop after applying a gold discount.")