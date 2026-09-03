import os
from dotenv import load_dotenv
from groq import Groq
import time
import re

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

model="qwen/qwen3.6-27b"

def check_product_price(product_name):
    if product_name == "laptop":
        return 1000
    elif product_name == "phone":
        return 3500
    else:
        return "Product not found"

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Invalid expression"

tools={
    "check_product_price": check_product_price,
    "calculator": calculator
}

system_prompt="""
    You are a Shopping Assistant Agent.

    Your job is to answer the user's shopping-related questions using the available tools when necessary.

    AVAILABLE TOOLS:

    1. check_product_price(product_name)
    - Returns the price of a product.
    - Available products: laptop, phone.

    2. calculator(expression)
    - Performs mathematical calculations.

    RULES:

    1. Decide whether a tool is required.
    2. If a tool is required, output exactly ONE action using this format:

    Action: tool_name("input")

    Examples:

    Action: check_product_price("laptop")

    Action: calculator("2000 - 1000")

    3. Do not output explanations, reasoning, thoughts, analysis, or rationale before the Action.
    4. Do not output <think>...</think>.
    5. Do not simulate tool results.
    6. Do not guess or invent tool results.
    7. Wait for the Observation from the Python program before taking the next action.
    8. After receiving an Observation, either:
    - output the next Action, or
    - output the Final Answer.
    9. If no available tool can handle the request, output exactly:

    Tool not found

    10. When the task is complete, output exactly:

    Final Answer: <answer>

    IMPORTANT:
    The Python program will execute the Action and provide the Observation.
    You only need to decide the next Action or provide the Final Answer.
    Do not reveal your internal reasoning.
"""

def react_agent(user_prompt):
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    while True:
        response=client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        answer=response.choices[0].message.content
        print(answer)

        if "Tool not found" in answer:
            break
        if "Final Answer" in answer:
            break
        
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name=match.group(1)
            tool_input=match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            if tool_name in tools:
                tool=tools[tool_name]
                observation=tool(tool_input)
            else:
                observation="Tool not found"

            print(f"Tool: {tool_name}")
            print(f"Input: {tool_input}")
            print(f"Observation: {observation}")
            print("-" * 50)


            messages.append({
                "role": "assistant",
                "content": answer
            })

            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

            time.sleep(5)



user_prompt="i want to buy iphone 17. i have 2000 rupees. if i buy a iphone 17 then how much money would i left up with?"

react_agent(user_prompt)