"""Load structured prompt sets from YAML files.

Each YAML file corresponds to a pipeline stage and contains named prompt sets.
Each prompt set has ``system``, ``user`` (or ``user_json``/``user_text``), and
optionally ``tool`` keys.

Public API:
    get_prompts(stage)            -> all prompt sets for a stage
    get_prompt_set(stage, name)   -> single prompt set dict
"""
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).parent

# Expected YAML files and the prompt sets each must contain.
# Every prompt set must have a 'system' key (except non-prompt entries
# like 'style_guidance' which are listed here as None).
_EXPECTED_SCHEMA: dict[str, dict[str, list[str] | None]] = {
    "clarify_query": {
        "generate_questions": ["system", "user"],
        "synthesize_brief": ["system", "user"],
    },
    "explore_topic": {
        "generate_planning_queries": ["system", "user", "tool"],
        "analyze_landscape": ["system", "user"],
        "analyze_page": ["system", "user"],
    },
    "design_outline": {
        "design_outline": ["system", "user"],
    },
    "plan_tasks": {
        "plan_tasks": ["system", "user"],
    },
    "research_topic": {
        "generate_queries": ["system", "user_json", "user_text", "tool"],
        "extract_source": ["system", "user"],
        "identify_gaps": ["system", "user"],
        "synthesize_notes": ["system", "user"],
    },
    "review_gaps": {
        "analyze_gaps": ["system", "user"],
    },
    "synthesize_sections": {
        "style_guidance": None,  # not a prompt set, just a data dict
        "synthesize_section": ["system", "user"],
    },
    "write_report": {
        "executive_summary": ["system", "user"],
        "conclusion": ["system", "user"],
    },
}

# stage_name -> dict of prompt sets (loaded once at import time)
_STAGE_DATA: dict = {}


def _load_all() -> None:
    """Load every .yaml file, keyed by filename stem. Errors are fatal."""
    for yaml_file in sorted(_PROMPTS_DIR.glob("*.yaml")):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            raise RuntimeError(
                f"Prompt file {yaml_file.name} is empty or failed to parse"
            )
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Prompt file {yaml_file.name} must be a YAML mapping, "
                f"got {type(data).__name__}"
            )
        _STAGE_DATA[yaml_file.stem] = data


def _validate_schema() -> None:
    """Validate all expected stages and prompt sets are loaded with required keys."""
    for stage, prompt_sets in _EXPECTED_SCHEMA.items():
        if stage not in _STAGE_DATA:
            raise RuntimeError(
                f"Required prompt stage {stage!r} not loaded. "
                f"Expected file: {stage}.yaml"
            )
        stage_data = _STAGE_DATA[stage]
        for set_name, required_keys in prompt_sets.items():
            if set_name not in stage_data:
                raise RuntimeError(
                    f"Missing prompt set {set_name!r} in {stage}.yaml. "
                    f"Available: {list(stage_data.keys())}"
                )
            if required_keys is None:
                continue  # non-prompt data entry, skip key validation
            entry = stage_data[set_name]
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Prompt set {stage!r}/{set_name!r} must be a mapping, "
                    f"got {type(entry).__name__}"
                )
            for key in required_keys:
                if key not in entry:
                    raise RuntimeError(
                        f"Prompt set {stage!r}/{set_name!r} missing "
                        f"required key {key!r}. Has: {list(entry.keys())}"
                    )
                value = entry[key]
                if key == "tool":
                    if not isinstance(value, dict):
                        raise RuntimeError(
                            f"Prompt set {stage!r}/{set_name!r} 'tool' must be "
                            f"a mapping, got {type(value).__name__}"
                        )
                else:
                    if not isinstance(value, str) or not value.strip():
                        raise RuntimeError(
                            f"Prompt set {stage!r}/{set_name!r} key {key!r} "
                            f"must be a non-empty string"
                        )


_load_all()
_validate_schema()


def get_prompts(stage: str) -> dict:
    """Return all prompt sets for a stage.

    Example: ``get_prompts("clarify_query")``
    """
    if stage not in _STAGE_DATA:
        raise KeyError(f"Unknown prompt stage: {stage!r}")
    return _STAGE_DATA[stage]


def get_prompt_set(stage: str, call_name: str) -> dict:
    """Return a single prompt set.

    Example: ``get_prompt_set("clarify_query", "generate_questions")``
    Returns dict with 'system', 'user', and optionally 'tool' keys.
    """
    stage_data = get_prompts(stage)
    if call_name not in stage_data:
        raise KeyError(
            f"Unknown prompt set {call_name!r} in stage {stage!r}. "
            f"Available: {list(stage_data.keys())}"
        )
    return stage_data[call_name]
