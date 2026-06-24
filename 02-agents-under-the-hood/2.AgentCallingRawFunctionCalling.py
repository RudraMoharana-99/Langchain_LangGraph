from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable


MAX_ITERATION = 10
MODEL = "qwen3:1.7b"


@traceable(name="Ollama tool for product price")
def get_product_price(product: str) -> float:
    """Look up a product price in catalog.."""
    print(f"    Executing get_product_price(product= {product})")
    prices = {"laptop": 1299.99, "headphones": 149.99, "keyboard": 145.45}

    return prices.get(product, 0)


@traceable(name="Ollama tool for apply discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
       Available tiers: bronze, silver and gold"""

    print(f"    Executing apply_discount(price: {price}, discount_tier: {discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1-discount/100), 2)


# Difference 2: Without @tool , we must manually create the JSON schema for each function
# This is exactly what the langchain @tool decorator generate automatically
# from the function's type hint and doc-string

tools_for_llm = [
    {
    "type":"function",
    "function":{
        "name": "get_product_price",
        "description": "Look up the price of product in the catalog",
        "parameters":{
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "The product name, e.g. 'laptop', 'headphone', 'keyboard'"
                },
            },
            "required": ["product"]
        },
    },
},
{
"type":"function",
"function":{
    "name": "apply_discount",
    "description": "Apply discount tier to a price and return the final price."
       "Available tiers: bronze, silver and gold",
    "parameters":{
        "type": "object",
        "properties": {
            "price": {"type": "number","description": "The original price"},
            "discount_tier": {"type": "string", "description": "Available tiers: bronze, silver and gold"},
        },
        "required": ["price", "discount_tier"]
    }
}
},

]

# Note: Ollama can also generate these schema if you pass the functions
# directly as tools( similar to langchian's @tool decorator)
# tools_for_llm = [get_product_price,apply_discount]
# however, this requires your function's docstring must follow Google's docstring format.

#   def get_product_price(product: str) -> float:
#       """Look up a product price in catalog
#          
#       Args:
#           product: The product name e.g. 'laptop', 'headphone', 'keyboard'
#       
#       Returns:
#           The price of the product or 0 if not found. 
# 
#       """

# We keep the mannula json version here, so we can know what @tool hide from us.

#---------@Helper traced ollama call---------------
# Difference 3: Without langchain we must manually trace LLM call for LangSmith

@traceable(name="ollama call", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)



@traceable(name="LangChain Agent Loop")
def run_agent(question: str) -> str:
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")

    messages = [
        {
            "role": "system",
            "content": (
                
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
                            
            ),
        },
        {
            "role": "user",
            "content": question
        },
    ]

    for iteration in range(1, MAX_ITERATION+1):
        print(f"\n--- Iteration : {iteration}---")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\Final answer: {ai_message.content}")
            return ai_message.content

        # Process only first tool call - force on tool per iteration
        tool_call = tool_calls[0]

        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tools_to_use = tools_dict.get(tool_name)

        if tools_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke
        observation = tools_to_use(**tool_args)

        print(f"[Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": str(observation),
            }
        )
    print("ERROR: max iteration reached without a final answer.")
    return

if __name__ == "__main__":
    print("Hello LangCahin Agent (.bind tool)")
    print()
    result = run_agent(question="What is the price of laptop after applying a gold discount.")