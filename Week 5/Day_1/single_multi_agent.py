"""Compare a 3-call single-agent workflow with a manager + 3-specialist workflow.

Provider usage fields (prompt_tokens, completion_tokens, total_tokens) are recorded
for every Groq Chat Completions request. Function-call arguments are part of the
model completion usage; tool results become input to later requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "qwen/qwen3.8-27b"
MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
MANAGER_MODEL = os.getenv("MANAGER_MODEL", MODEL)
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", MODEL)
ANALYST_MODEL = os.getenv("ANALYST_MODEL", MODEL)
RECOMMENDER_MODEL = os.getenv("RECOMMENDER_MODEL", MODEL)
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "700"))

if not API_KEY:
    raise SystemExit(
        "GROQ_API_KEY is missing. Add it to .env, for example:\n"
        'GROQ_API_KEY="gsk_your_key_here"'
    )

client = Groq(api_key=API_KEY)


@dataclass
class UsageRow:
    mode: str
    call_number: int
    role: str
    model: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None


USAGE_ROWS: list[UsageRow] = []


def _get_int(obj: Any, field: str, default: int = 0) -> int:
    value = getattr(obj, field, default)
    return value if isinstance(value, int) else default


def call_llm(
    *,
    mode: str,
    role: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> Any:
    """Make one API request and record provider-reported token usage."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    # gpt-oss on Groq supports reasoning_effort. Keep all roles consistent.
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    response = client.chat.completions.create(**kwargs)
    usage = response.usage
    detail = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = None
    if detail is not None:
        value = getattr(detail, "reasoning_tokens", None)
        if isinstance(value, int):
            reasoning_tokens = value

    row = UsageRow(
        mode=mode,
        call_number=1 + sum(1 for item in USAGE_ROWS if item.mode == mode),
        role=role,
        model=model,
        prompt_tokens=_get_int(usage, "prompt_tokens"),
        output_tokens=_get_int(usage, "completion_tokens"),
        total_tokens=_get_int(usage, "total_tokens"),
        reasoning_tokens=reasoning_tokens,
    )
    USAGE_ROWS.append(row)
    reasoning_text = (
        str(row.reasoning_tokens) if row.reasoning_tokens is not None else "n/a"
    )
    print(
        f"[TOKENS] {mode:6} | call {row.call_number:02} | {role:24} | "
        f"input={row.prompt_tokens:5} | output={row.output_tokens:5} | "
        f"total={row.total_tokens:5} | reasoning={reasoning_text}"
    )
    return response


def print_token_summary(mode: str) -> None:
    rows = [row for row in USAGE_ROWS if row.mode == mode]
    input_total = sum(row.prompt_tokens for row in rows)
    output_total = sum(row.output_tokens for row in rows)
    total_total = sum(row.total_tokens for row in rows)
    reasoning_rows = [row.reasoning_tokens for row in rows if row.reasoning_tokens is not None]
    reasoning_total = sum(reasoning_rows) if reasoning_rows else None

    print(f"\n{'=' * 80}")
    print(f"{mode.upper()} TOKEN SUMMARY")
    print(f"{'=' * 80}")
    print(f"LLM API calls       : {len(rows)}")
    print(f"Input/prompt tokens : {input_total}")
    print(f"Output/completion   : {output_total}")
    print(f"Total tokens        : {total_total}")
    if reasoning_total is not None:
        print(f"Reasoning tokens*   : {reasoning_total}")
    print("*Reasoning tokens are shown separately when the API returns that field.")


SINGLE_SYSTEM = """You are one helpful AI Engineering assistant executing a 3-stage workflow.
Complete only the stage requested in the current call. Be concise but useful.
Do not invent benchmark measurements. Explain token/call overhead conceptually;
actual token measurements will be printed by the Python program."""

SINGLE_TASKS = [
    "Stage 1/3 — Explain single-agent and multi-agent LLM systems, their basic architecture, strengths, limitations, and typical use cases. Keep the result structured and concise.",
    "Stage 2/3 — Analyze token/call overhead, repeated context, latency, coordination, and failure modes for single-agent vs multi-agent systems. Use the Stage 1 notes below. Do not make up numeric token counts.\n\nSTAGE 1 NOTES:\n{stage1}",
    "Stage 3/3 — Produce the final concise comparison for an AI Engineering class. Include a compact comparison table and a practical rule for choosing each architecture. Incorporate the Stage 1 and Stage 2 notes below. Do not invent measured numbers; the program reports actual API usage separately.\n\nSTAGE 1 NOTES:\n{stage1}\n\nSTAGE 2 NOTES:\n{stage2}",
]


def run_single() -> str:
    print("\n\n########## SINGLE AGENT: SAME LLM, 3 CALLS ##########")
    outputs: list[str] = []
    for index, template in enumerate(SINGLE_TASKS):
        prompt = template.format(
            stage1=outputs[0] if len(outputs) > 0 else "",
            stage2=outputs[1] if len(outputs) > 1 else "",
        )
        response = call_llm(
            mode="single",
            role=f"single_agent_stage_{index + 1}",
            model=MODEL,
            messages=[
                {"role": "system", "content": SINGLE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        outputs.append(content)
        print(f"\n--- Single Agent / Stage {index + 1} ---\n{content}\n")

    print_token_summary("single")
    return outputs[-1]


# Manager tool schemas: each tool delegates to one specialist LLM. The local
# Python orchestration executes the specialist call and returns its result.
def make_delegate_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear instructions to the specialist for this stage.",
                    }
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    }


DELEGATE_TOOLS = {
    "delegate_to_researcher": make_delegate_tool(
        "delegate_to_researcher",
        "Delegate system definitions, architecture, strengths, limitations, and use cases to the researcher specialist.",
    ),
    "delegate_to_analyst": make_delegate_tool(
        "delegate_to_analyst",
        "Delegate token/call overhead, latency, context duplication, coordination, and failure-mode analysis to the analyst specialist.",
    ),
    "delegate_to_recommender": make_delegate_tool(
        "delegate_to_recommender",
        "Delegate practical selection criteria and recommendations to the recommender specialist, using prior specialist findings.",
    ),
}

MANAGER_MODEL_SYSTEM = """You are the manager agent coordinating three specialist LLMs.
The user has asked for a grounded comparison of single-agent and multi-agent systems.
At each stage the application will expose exactly one specialist-delegation tool and
force that tool to be called. Write a useful, specific 'task' instruction for that
specialist. Use earlier specialist outputs available in the conversation when relevant.
Do not invent token benchmark measurements; the Python harness measures actual usage."""

MANAGER_FINAL_SYSTEM = """You are the manager agent. The three specialists have completed their tasks.
Synthesize their findings into a clear, concise AI Engineering class report comparing
single-agent and multi-agent LLM systems. Include a compact table, benefits/limitations,
and a practical rule for choosing each. Do not claim the token counts in this report;
the Python harness prints the measured counts after the run. If the specialists disagree,
explain the trade-off briefly."""

SPECIALISTS = [
    {
        "tool_name": "delegate_to_researcher",
        "role": "researcher",
        "model": RESEARCHER_MODEL,
        "system": "You are the researcher specialist. Explain concepts and architecture accurately, compare strengths and limitations, and keep the output concise.",
        "default_task": "Explain single-agent and multi-agent LLM systems, their architectures, strengths, limitations, and typical use cases. Keep it concise.",
    },
    {
        "tool_name": "delegate_to_analyst",
        "role": "analyst",
        "model": ANALYST_MODEL,
        "system": "You are the analysis specialist. Compare execution calls, input/output token drivers, repeated context, latency, coordination overhead, and failure modes. Never invent numerical benchmark results.",
        "default_task": "Using the available researcher notes, compare token and API-call overhead, latency, context duplication, coordination, and failure modes. Do not invent numeric measurements.",
    },
    {
        "tool_name": "delegate_to_recommender",
        "role": "recommender",
        "model": RECOMMENDER_MODEL,
        "system": "You are the recommendation specialist. Convert the available technical findings into practical architecture-selection criteria and a realistic example.",
        "default_task": "Using the previous findings, recommend when to use a single agent and when to use a manager-plus-specialists multi-agent system. Include one realistic example.",
    },
]

MULTI_USER_REQUEST = """Prepare a concise AI Engineering class comparison of single-agent and multi-agent LLM systems.
The final deliverable should cover: (1) definitions and architecture, (2) trade-offs involving
output/input tokens, API calls, latency, context duplication, and coordination, and (3) a practical
recommendation for choosing between the architectures. Three specialist stages should cover
research, analysis, and recommendation, in that order."""


def run_multi() -> str:
    print("\n\n########## MULTI AGENT: MANAGER + 3 SPECIALISTS ##########")
    manager_messages: list[dict[str, Any]] = [
        {"role": "user", "content": MULTI_USER_REQUEST}
    ]
    specialist_results: dict[str, str] = {}

    for specialist in SPECIALISTS:
        tool_name = specialist["tool_name"]
        tool = DELEGATE_TOOLS[tool_name]

        # Force the manager to use the expected delegation tool for this stage.
        # The model still writes the actual task instructions in the tool arguments.
        dispatch_response = call_llm(
            mode="multi",
            role=f"manager_dispatch_{specialist['role']}",
            model=MANAGER_MODEL,
            messages=[
                {"role": "system", "content": MANAGER_MODEL_SYSTEM},
                *manager_messages,
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        manager_message = dispatch_response.choices[0].message
        tool_calls = manager_message.tool_calls or []
        if not tool_calls:
            raise RuntimeError(
                f"Manager did not return the forced tool call '{tool_name}'. "
                "Check the selected Groq model and tool-calling support."
            )

        tool_call = tool_calls[0]
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
            delegated_task = str(arguments.get("task", "")).strip()
        except (json.JSONDecodeError, AttributeError, TypeError):
            delegated_task = ""
        if not delegated_task:
            delegated_task = specialist["default_task"]

        # Keep a valid tool-call transcript so the manager receives the specialist result.
        manager_messages.append(
            {
                "role": "assistant",
                "content": manager_message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
        )

        context = ""
        if specialist_results:
            context = "\n\nPREVIOUS SPECIALIST RESULTS:\n" + json.dumps(
                specialist_results, ensure_ascii=False, indent=2
            )
        specialist_user_prompt = (
            f"MANAGER-ASSIGNED TASK:\n{delegated_task}\n\n"
            "Overall objective: compare single-agent and multi-agent LLM systems for an AI Engineering class. "
            "Be concise, accurate, and do not invent measured token counts."
            f"{context}"
        )

        specialist_response = call_llm(
            mode="multi",
            role=f"specialist_{specialist['role']}",
            model=specialist["model"],
            messages=[
                {"role": "system", "content": specialist["system"]},
                {"role": "user", "content": specialist_user_prompt},
            ],
        )
        result = specialist_response.choices[0].message.content or ""
        specialist_results[specialist["role"]] = result
        print(f"\n--- Multi Agent / {specialist['role'].title()} ---\n{result}\n")

        manager_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": result,
            }
        )

    # The manager now has all three tool results and produces the final synthesis.
    final_response = call_llm(
        mode="multi",
        role="manager_final_synthesis",
        model=MANAGER_MODEL,
        messages=[
            {"role": "system", "content": MANAGER_FINAL_SYSTEM},
            *manager_messages,
        ],
    )
    final_answer = final_response.choices[0].message.content or ""
    print(f"\n--- Multi Agent / Final Manager Report ---\n{final_answer}\n")
    print_token_summary("multi")
    return final_answer


def print_comparison() -> None:
    totals: dict[str, dict[str, int]] = {}
    for mode in ("single", "multi"):
        rows = [row for row in USAGE_ROWS if row.mode == mode]
        totals[mode] = {
            "calls": len(rows),
            "input": sum(row.prompt_tokens for row in rows),
            "output": sum(row.output_tokens for row in rows),
            "total": sum(row.total_tokens for row in rows),
        }

    single_output = totals["single"]["output"]
    multi_output = totals["multi"]["output"]
    delta = multi_output - single_output
    percentage = (delta / single_output * 100) if single_output else 0.0

    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON: PROVIDER-REPORTED TOKEN USAGE")
    print("=" * 80)
    print(f"{'Metric':<24}{'Single agent':>18}{'Multi agent':>18}{'Difference':>18}")
    print("-" * 78)
    for key, label in [("calls", "LLM API calls"), ("input", "Input tokens"), ("output", "Output tokens"), ("total", "Total tokens")]:
        diff = totals["multi"][key] - totals["single"][key]
        print(
            f"{label:<24}{totals['single'][key]:>18}{totals['multi'][key]:>18}{diff:>+18}"
        )
    print(f"\nMulti-agent output-token change vs single-agent: {delta:+} tokens ({percentage:+.1f}%).")
    print(
        "Note: output tokens are the API's completion_tokens for every call, including manager-generated "
        "tool-call arguments. Tool-result text is counted as input in later calls, not as model output."
    )
    print(
        "For a fairer model comparison, all roles default to the same model ID. You can override role models "
        "in .env, but then architecture and model differences are mixed together."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-agent and/or manager+specialists token comparison on Groq."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "both"],
        default="both",
        help="Which architecture to run (default: both).",
    )
    args = parser.parse_args()

    if args.mode == "single":
        run_single()
    elif args.mode == "multi":
        run_multi()
    else:
        run_single()
        run_multi()
        print_comparison()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Check GROQ_API_KEY, model access, current Groq limits, and that the chosen model supports tool calling.",
            file=sys.stderr,
        )
        raise
