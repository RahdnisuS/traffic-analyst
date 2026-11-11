"""Deploy the demo to Google Cloud Run: ``python -m deploy.cloudrun <project-id> [--region us-central1]``.

WHY Cloud Run (decision log, Phase 14): Hugging Face started charging for Docker Spaces; Render's
free tier has 512 MB of RAM, too little for PyTorch + the search models. Cloud Run runs the same
Dockerfile, scales to zero when nobody visits (cost ~0 within the free tier) and supports WebSockets.

Needs the gcloud CLI, logged in (``gcloud auth login``), and a project with billing linked. Steps:
  1. enable the APIs it needs (Run, Cloud Build, Artifact Registry, Secret Manager);
  2. store GROQ_API_KEY from the local .env in Secret Manager (encrypted; the container gets it as an
     environment variable; it is never in the image, the source upload or the build log) and let
     the service's account read it;
  3. ``gcloud run deploy --source .``: Cloud Build builds the Dockerfile from the files .gcloudignore
     lets through (never .env or any PDF), then the service starts.
Settings, and why:
  --max-instances 1   sessions live in one process's memory; a second instance would split them
                      (and it caps the bill)
  --min-instances 0   scale to zero: no visitors, no cost (the next visitor waits ~30 s to start)
  --memory 2Gi        PyTorch + two search models + the index + up to 4 sims (+ in-memory disk)
  --cpu 1             the free tier is CPU-seconds: 1 vCPU doubles the free hours vs 2
  --timeout 3600      a WebSocket is one long request; Cloud Run's maximum is 60 min, after which
                      the page reconnects to the same session by itself
  --session-affinity  a reconnecting tab reaches the same instance
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from rag.llm import load_api_key
from sim.config import PROJECT_ROOT

SERVICE = "traffic-analyst"
SECRET = "groq-api-key"


def gcloud(*args: str, input: str | None = None, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    exe = shutil.which("gcloud") or sys.exit("gcloud is not installed: https://cloud.google.com/sdk/docs/install")
    if not quiet:
        print("$ gcloud " + " ".join(args))
    return subprocess.run([exe, *args], input=input, text=True, check=check, cwd=PROJECT_ROOT,
                          capture_output=quiet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project", help="Google Cloud project id")
    ap.add_argument("--region", default="us-central1",
                    help="a tier-1 region keeps the free tier's prices (default us-central1)")
    ap.add_argument("--no-secret", action="store_true", help="deploy without the analyst's key")
    args = ap.parse_args()
    p = ["--project", args.project]

    gcloud("services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com",
           "artifactregistry.googleapis.com", "secretmanager.googleapis.com", *p)
    secret_flags = []
    if not args.no_secret:
        key = load_api_key() or sys.exit("no GROQ_API_KEY in .env (or pass --no-secret)")
        exists = gcloud("secrets", "describe", SECRET, *p, check=False, quiet=True).returncode == 0
        if exists:
            gcloud("secrets", "versions", "add", SECRET, "--data-file=-", *p, input=key)
        else:
            gcloud("secrets", "create", SECRET, "--replication-policy=automatic", "--data-file=-", *p, input=key)
        number = gcloud("projects", "describe", args.project, "--format=value(projectNumber)",
                        quiet=True).stdout.strip()
        gcloud("secrets", "add-iam-policy-binding", SECRET, *p,
               f"--member=serviceAccount:{number}-compute@developer.gserviceaccount.com",
               "--role=roles/secretmanager.secretAccessor", "--condition=None", quiet=True)
        secret_flags = [f"--set-secrets=GROQ_API_KEY={SECRET}:latest"]
    gcloud("run", "deploy", SERVICE, "--source", ".", "--region", args.region, *p,
           "--allow-unauthenticated", "--max-instances=1", "--min-instances=0", "--memory=2Gi", "--cpu=1",
           "--timeout=3600", "--session-affinity", "--cpu-boost", *secret_flags)
    url = gcloud("run", "services", "describe", SERVICE, "--region", args.region, *p,
                 "--format=value(status.url)", quiet=True).stdout.strip()
    print(f"\nLive at {url}")


if __name__ == "__main__":
    main()
