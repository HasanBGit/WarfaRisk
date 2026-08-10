"""Push trained model artifacts to Hugging Face Hub, with the same
verification discipline as RESULTS.md: a push isn't "done" until it's been
independently confirmed to exist on the Hub, not just because the upload
call returned without raising.

Naming convention: every repo created from this project is prefixed
`warfarin-review-` so it's distinguishable from the account's other
(unrelated) projects. Auth comes from whatever `huggingface_hub` already
has cached (`hf auth login`) or an `HF_TOKEN` env var / `.env` entry — this
module never asks for or stores a token itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

REPO_PREFIX = "warfarin-review-"


def push_model_dir(
    repo_name: str,
    local_dir: Path,
    commit_message: str,
    private: bool = True,
) -> str:
    """Uploads every file in local_dir to a model repo named
    f"{REPO_PREFIX}{repo_name}" under whichever account the cached token
    belongs to. Returns the repo_id actually used.

    private=True by default — this is unpublished research output, not a
    public release; flip explicitly (and deliberately) when ready to publish.
    """
    api = HfApi()
    who = api.whoami()
    repo_id = f"{who['name']}/{REPO_PREFIX}{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir), commit_message=commit_message)
    return repo_id


def verify_pushed(repo_id: str) -> dict:
    """Confirms a repo actually exists on the Hub and lists its files —
    call this after every push and only record the result as fact if this
    succeeds. Never trust upload_folder()'s return value alone; a prior
    project's claimed HF push turned out, on this exact check, to have
    never happened."""
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    return {"repo_id": repo_id, "exists": True, "siblings": [s.rfilename for s in info.siblings], "sha": info.sha}


def write_model_card(out_path: Path, title: str, body_md: str) -> None:
    out_path.write_text(f"# {title}\n\n{body_md}\n")


def write_metrics_json(out_path: Path, metrics: dict) -> None:
    out_path.write_text(json.dumps(metrics, indent=2))
