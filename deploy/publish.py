"""Publish the demo to a Hugging Face Space: ``python -m deploy.publish <user>/<space>``.

Needs a one-time ``hf auth login`` (a token with write access). What it does:
  1. creates the Space (Docker SDK) if it doesn't exist;
  2. uploads exactly the files git tracks (so .env, local run logs and the owner's copyrighted
     PDF can never be uploaded by accident), with deploy/space_README.md as the Space's README
     (Hugging Face reads the Space settings from that file's front matter);
  3. stores GROQ_API_KEY from the local .env as a Space SECRET (encrypted, given to the container
     as an environment variable, never visible in the files or the build log).
Hugging Face then builds the Dockerfile and starts the container (about 10 minutes the first time).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from rag.llm import load_api_key
from sim.config import PROJECT_ROOT

NEVER = (".env", ".claude/", "data/", "chroma/", "eval/work/")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=PROJECT_ROOT, capture_output=True, check=True).stdout
    files = [f for f in out.decode().split("\0") if f]
    bad = [f for f in files if f.startswith(NEVER) or f.lower().endswith(".pdf")]
    if bad:  # belt and braces: these must never leave the machine, whatever git tracks
        sys.exit(f"refusing to publish, these files must not be uploaded: {bad}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("space", help="<hf-username>/<space-name>, e.g. jdoe/traffic-analyst")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--no-secret", action="store_true", help="don't set GROQ_API_KEY (analyst disabled)")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.whoami()  # fails early with a clear message if not logged in
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    files = [f for f in tracked_files() if f != "README.md"]
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(PROJECT_ROOT / f)) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=str(PROJECT_ROOT / "deploy" / "space_README.md")))
    # Files deleted locally since the last publish are removed from the Space too.
    remote = set(api.list_repo_files(args.space, repo_type="space"))
    stale = sorted(remote - set(files) - {"README.md", ".gitattributes"})
    if stale:
        from huggingface_hub import CommitOperationDelete
        ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
    commit = api.create_commit(args.space, repo_type="space", operations=ops,
                               commit_message="Publish Traffic Analyst demo")
    print(f"uploaded {len(files) + 1} files ({len(stale)} removed): {commit.commit_url}")
    if not args.no_secret:
        key = load_api_key()
        if key:
            api.add_space_secret(args.space, "GROQ_API_KEY", key,
                                 description="Groq API key for the analyst (free tier)")
            print("GROQ_API_KEY set as a Space secret")
        else:
            print("no GROQ_API_KEY in .env: the analyst tab will say it is unavailable")
    print(f"building: https://huggingface.co/spaces/{args.space}  (logs: the 'Logs' button there)")


if __name__ == "__main__":
    main()
