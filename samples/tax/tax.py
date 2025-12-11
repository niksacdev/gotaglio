from glom import glom
import json
import os
from rich.console import Console
from rich.text import Text
import sys
from typing import Any

# Add the parent directory to the sys.path so that we can import from the
# gotaglio package, as if it had been installed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gotaglio.dag import Dag
from gotaglio.exceptions import ExceptionContext
from gotaglio.format import format_messages
from gotaglio.main import main
from gotaglio.pipeline_spec import (
    FormatterSpec,
    get_result,
    get_stages,
    get_turn,
    PipelineSpec,
    SummarizerSpec,
    column_spec,
)
from gotaglio.pipeline import Internal, Prompt
from gotaglio.shared import build_template, to_json_string
from gotaglio.summarize import keywords_column


###############################################################################
#
# Default Configuration Values
#
###############################################################################
configuration = {
    "prepare": {
        "template": Prompt("Template file for system message"),
        "template_text": Internal(),
    },
    "infer": {
        "model": {
            "name": Prompt("Model name to use for inference stage"),
            "settings": {
                "max_tokens": 1000,
                "temperature": 0.2,  # Lower temp for factual tax answers
            },
        }
    },
}


###############################################################################
#
# Stage Functions
#
###############################################################################
def stages(name, config, registry):
    """
    Defines the tax Q&A pipeline with four stages:
      **prepare** - creates the system prompt and user question
      **infer** - invokes the model to generate a response
      **extract** - extracts JSON from the model response
      **assess** - compares extracted answer to expected answer
    """
    template = build_template(
        config,
        "prepare.template",
        "prepare.template_text",
    )

    model = registry.model(glom(config, "infer.model.name"))

    async def prepare(context):
        """Build messages for the LLM."""
        turn_index = len(context["turns"]) - 1
        turn = context["case"]["turns"][turn_index]

        system = {"role": "system", "content": await template(context)}
        user = {"role": "user", "content": turn["user"]}

        return [system, user]

    async def infer(context):
        """Call the model."""
        stages = get_stages(context)
        return await model.infer(stages["prepare"], context)

    async def extract(context):
        """Parse JSON from LLM response."""
        stages = get_stages(context)
        with ExceptionContext("Extracting JSON from LLM response."):
            text = stages["infer"]

            # Debug: print raw response if empty or problematic
            if not text or not text.strip():
                raise ValueError(f"LLM returned empty response")

            # Strip fenced code block markers if present
            if text.startswith("```json"):
                text = text[7:]  # len("```json")
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            if not text:
                raise ValueError(f"Response was only code fences, no content")

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                # Show what we tried to parse
                preview = text[:200] + "..." if len(text) > 200 else text
                raise ValueError(f"Invalid JSON. Response was: {preview}") from e

    async def assess(context):
        """Compare extracted to expected."""
        stages = get_stages(context)
        turn = get_turn(context)

        extracted = stages["extract"]
        expected = turn["expected"]

        errors = []

        # Check taxability
        if extracted.get("taxable") != expected.get("taxable"):
            errors.append({
                "field": "taxable",
                "expected": expected.get("taxable"),
                "actual": extracted.get("taxable"),
            })

        # Check rate (with tolerance)
        ext_rate = extracted.get("rate")
        exp_rate = expected.get("rate")
        if exp_rate is not None and ext_rate is not None:
            if abs(ext_rate - exp_rate) > 0.01:  # 1% tolerance
                errors.append({
                    "field": "rate",
                    "expected": exp_rate,
                    "actual": ext_rate,
                    "variance": abs(ext_rate - exp_rate),
                })

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "cost": len(errors),  # Cost = number of errors
        }

    return Dag.from_linear({
        "prepare": prepare,
        "infer": infer,
        "extract": extract,
        "assess": assess,
    })


###############################################################################
#
# Summarizer extensions
#
###############################################################################
def taxable_cell(result: dict[str, Any], turn_index: int):
    """Show taxability result in summary."""
    extracted = glom(get_stages(result, turn_index), "extract.taxable", default=None)
    expected = glom(get_turn(result, turn_index), "expected.taxable", default=None)

    if extracted is None:
        return Text("?", style="yellow")

    match = extracted == expected
    text = "Yes" if extracted else "No"
    return Text(text, style="bold green" if match else "bold red")


def rate_cell(result: dict[str, Any], turn_index: int):
    """Show rate result in summary (raw value from LLM)."""
    extracted = glom(get_stages(result, turn_index), "extract.rate", default=None)

    if extracted is None:
        return Text("-")

    # Show raw value - if LLM returns 6.25 instead of 0.0625, you'll see it
    return Text(f"{extracted}")


def question_cell(result, turn_index):
    """Show the question in summary."""
    question = get_turn(result, turn_index)["user"]
    # Truncate if too long
    if len(question) > 60:
        question = question[:57] + "..."
    return question


###############################################################################
#
# Formatter extensions
#
###############################################################################
def format_turn(console: Console, turn_index, result: dict[str, Any]):
    """Format a single turn for detailed output."""
    stages = get_result(result, turn_index)
    passed = passed_predicate(result, turn_index)
    turn = result["case"]["turns"][turn_index]

    if passed:
        console.print(f"### Turn {turn_index + 1}: **PASSED**  ")
    else:
        errors = glom(stages, "stages.assess.errors", default=[])
        console.print(f"### Turn {turn_index + 1}: **FAILED** ({len(errors)} errors)  ")

    console.print()
    console.print(f"**Question:** {turn['user']}")
    console.print()

    format_messages(console, stages["stages"]["prepare"], collapse=["system"])

    console.print("**Response:**")
    console.print("```json")
    console.print(to_json_string(stages["stages"]["extract"]))
    console.print("```")
    console.print()

    if not passed:
        console.print("**Expected:**")
        console.print("```json")
        console.print(to_json_string(turn["expected"]))
        console.print("```")
        console.print()
        console.print("**Errors:**")
        for error in stages["stages"]["assess"]["errors"]:
            console.print(f"* {error['field']}: expected={error['expected']}, actual={error['actual']}")


###############################################################################
#
# Pipeline extensions
#
###############################################################################
def expected(result, turn_index=None):
    """Returns the expected value from a turn."""
    return get_turn(result, turn_index)["expected"]


def passed_predicate(result, turn_index=None):
    """Predicate to determine if result is passing."""
    return glom(get_stages(result, turn_index), "assess.passed", default=False)


###############################################################################
#
# Pipeline specification
#
###############################################################################
tax_pipeline_spec = PipelineSpec(
    name="tax",
    description="US indirect tax Q&A evaluation pipeline",
    configuration=configuration,
    create_dag=stages,
    expected=expected,
    formatter=FormatterSpec(
        format_turn=format_turn,
    ),
    passed_predicate=passed_predicate,
    summarizer=SummarizerSpec(
        columns=[
            column_spec(name="taxable", contents=taxable_cell),
            column_spec(name="rate", contents=rate_cell),
            keywords_column,
            column_spec(name="question", contents=question_cell),
        ]
    ),
)


def go():
    main([tax_pipeline_spec])


if __name__ == "__main__":
    go()
