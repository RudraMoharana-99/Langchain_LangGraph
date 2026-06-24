import re
import inspect
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
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1-discount/100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# Change 3: Delete the JSON schemas. Tools now live inside the prompt as plain text
# we derive descriptions from the functions themselves using "Inspect library"

def get_tool_descriptions(tools_dict: dict):
    description = []

    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @tracable adds *, config=None)
        originl_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(originl_function)
        docstring = inspect.getdoc(tool_function) or ""
        description.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(description)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
            STRICT RULE - you must follow this exactly:
            1. Never guess or assume any product price. You must call get_product_price first to get the real price.
            2. Only call apply_discount AFTER you recieved a price from get_product_price. Pass the exact price
               returned by get_product_price - do NOT pass a made-up number.
            3. Never calculate discount yourself using math, Always use the apply_discount tool.
            4. If the user doesn't specify discount tier, ask them which tier to use - do NOT assume one.

            Answer the following questions as best you can, You have acess to the foloowing tools:

            {tool_descriptions}

            Use the following format:

            Question: the input question you must answer
            Thought: youshould always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action, as comma separated values
            Observation: the result of action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Final Answer: the final answer to the original input question

            Begin!

            Question: {{question}}
            Thought:"""

# CHANGE 4: Drop tools = from ollama.chat(). The LLM has no idea it's an agent -
# all agency comes from the prompt above and our regex parsing below

@traceable(name="ollama call", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)



@traceable(name="Prompt Loop RAW")
def run_agent(question: str) -> str:

    print(f"Question: {question}")
    print("="*60)

    # Change 5 : One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scartchpad = ""
    
    
    for iteration in range(1, MAX_ITERATION+1):
        print(f"\n--- Iteration : {iteration}---")

        full_prompt = prompt + scartchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.content
        print(f"LLM Output: \n{output}")

        print(f" [Parsing] Looking for the final answer in LLM output....")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f" [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # CHANGE 6: Parse tool calls from the raw text with regex - fragile if LLM doesn't follow format.
        print(f"    [Parsing]  Looking for action and action input in LLM output ...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match= re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(" [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"    [TOOL SELECTED] {tool_name} with args: {tool_input_raw}")

        # Split comma separated args: strip key=prefix if llm output key=value format
        raw_args= [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip("'\"") for x in raw_args]

        print(f"    [Tool Executing] {tool_name}({args})....")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not fount. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"    [Tool Result] {observation}")

        # Change 7: History is one growing string re-sent every iteration (replaces messages.append)
        scartchpad += f"{output}\nObservation: {observation}\nThought:"
        

        
    print("ERROR: max iteration reached without a final answer.")
    return

if __name__ == "__main__":
    print("Hello LangCahin Agent (.bind tool)")
    print()
    result = run_agent(question="What is the price of laptop after applying a gold discount.")