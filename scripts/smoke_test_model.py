#!/usr/bin/env python3
"""Send one minimal request to verify endpoint access and JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_model_config import load_env_file, resolved_config, validate

from ai_historian.model_config import create_chat_completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    load_env_file(args.env_file)
    config = resolved_config()
    errors, _ = validate(config)
    if errors:
        raise SystemExit("Configuration check failed: " + "; ".join(errors))

    from openai import OpenAI

    client = OpenAI(
        api_key=config["api_key"] or "EMPTY",
        base_url=config["base_url"],
        timeout=60,
    )
    kwargs = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "Return one valid JSON object."},
            {"role": "user", "content": 'Return exactly this meaning as JSON: status is ready.'},
        ],
        "max_tokens": 80,
    }
    kwargs["response_format"] = {"type": "json_object"}
    response = create_chat_completion(client, **kwargs)
    text = response.choices[0].message.content or ""
    payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
    print(json.dumps({"ok": True, "model": config["model"], "response": payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
