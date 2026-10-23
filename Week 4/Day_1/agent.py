import ast
import json
import math
import operator
import os

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# =====================================================
# 1. INITIALIZE CLIENTS
# =====================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing from .env")

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "qwen/qwen3.8-27b"


# =====================================================
# 2. CALCULATOR TOOL
# =====================================================

# Only these mathematical operators are allowed.
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(node):
    """Safely evaluate supported mathematical AST nodes."""

    if isinstance(node, ast.Expression):
        return evaluate_expression(node.body)

    # Allow integer and floating-point constants only.
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numbers are allowed.")
        return node.value

    # Handle expressions such as 10 + 5 or 2 ** 3.
    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in OPERATORS:
            raise ValueError("Unsupported mathematical operator.")

        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)

        # Prevent excessively large powers.
        if operator_type == ast.Pow and abs(right) > 12:
            raise ValueError("Exponent must be between -12 and 12.")

        result = OPERATORS[operator_type](left, right)

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Result is not a finite number.")

        return result

    # Handle expressions such as -5 and +10.
    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in UNARY_OPERATORS:
            raise ValueError("Unsupported unary operator.")

        return UNARY_OPERATORS[operator_type](
            evaluate_expression(node.operand)
        )

    # Reject function calls, variables, imports, etc.
    raise ValueError("Invalid mathematical expression.")


def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        if not expression.strip() or len(expression) > 200:
            raise ValueError("Expression must be 1-200 characters.")

        syntax_tree = ast.parse(expression, mode="eval")
        result = evaluate_expression(syntax_tree)

        return json.dumps({
            "expression": expression,
            "result": result,
        })

    except Exception as error:
        return json.dumps({
            "error": str(error),
        })


# =====================================================
# 3. WEB SEARCH TOOL
# =====================================================

def web_search(query: str) -> str:
    """Search the web using Tavily and return relevant sources."""

    try:
        if not query.strip() or len(query) > 400:
            raise ValueError("Query must be 1-400 characters.")

        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        # Return only the useful information to the LLM.
        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": (item.get("content") or "")[:1200],
            })

        return json.dumps({
            "answer": response.get("answer"),
            "results": results,
        }, ensure_ascii=False)

    except Exception as error:
        return json.dumps({
            "error": str(error),
        })


# =====================================================
# 4. STANDARD TOOL DEFINITIONS / SCHEMAS
# =====================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression. "
                "Use this tool for arithmetic calculations "
                "instead of calculating mentally. "
                "Supports +, -, *, /, //, %, ** and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression, such as "
                            "'125 * 48' or '(100 + 50) / 3'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for recent information, "
                "news, facts, websites, or information that "
                "requires external sources. Use this tool when "
                "the question needs web research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A clear search query describing "
                            "the information to find online."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# =====================================================
# 5. TOOL NAME -> PYTHON FUNCTION MAPPING
# =====================================================

AVAILABLE_TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
}


# =====================================================
# 6. AGENT LOOP / ORCHESTRATION
# =====================================================

def run_agent(user_query: str) -> str:
    """Run the LLM and execute tools until a final answer is ready."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI agent. "
                "You have access to a calculator and a web search tool. "
                "Use the calculator for mathematical calculations. "
                "Use web search for current information and external facts. "
                "You may call multiple tools when necessary. "
                "After receiving tool results, interpret them and provide "
                "a clear, accurate final answer. "
                "When using web search results, include relevant source URLs. "
                "Do not claim that a tool was used unless it was called."
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    # Limit iterations to avoid an endless tool-calling loop.
    for _ in range(6):

        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        # No more tool calls means the agent has a final response.
        if not tool_calls:
            return assistant_message.content or "No response generated."

        # Preserve the assistant's tool-call request in conversation history.
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        })

        # Execute every tool requested by the LLM.
        for call in tool_calls:
            tool_name = call.function.name

            try:
                tool_arguments = json.loads(
                    call.function.arguments or "{}"
                )

                tool_function = AVAILABLE_TOOLS.get(tool_name)

                if tool_function is None:
                    raise ValueError(f"Unknown tool: {tool_name}")

                print(f"\n[TOOL CALL] {tool_name}")
                print(f"[ARGUMENTS] {call.function.arguments}")

                tool_result = tool_function(**tool_arguments)

                print(f"[TOOL RESULT] {tool_result[:500]}")

            except Exception as error:
                tool_result = json.dumps({
                    "error": str(error),
                })

            # Send the tool result back to the LLM.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": str(tool_result),
            })

    return "Agent stopped after reaching the tool-call limit."


# =====================================================
# 7. RUN THE AGENT
# =====================================================

if __name__ == "__main__":
    print("Simple AI Agent is ready!")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in ("exit", "quit"):
            break

        if not query:
            continue

        try:
            answer = run_agent(query)
            print(f"\nAgent: {answer}")

        except Exception as error:
            print(f"\nError: {error}")