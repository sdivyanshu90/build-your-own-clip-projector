#!/usr/bin/env bash
# Commit and push every current project change as one-file commits.
# Run from the repository root with: bash scripts/commit-each-file.sh
#
# This is intentionally verbose for learning purposes. Read the function below:
#   1. git add -- <file> stages only that exact path.
#   2. git commit creates one commit from the staged path.
#   3. git push origin HEAD sends that one commit to the current branch on origin.
# The script exits on the first failure, so a failed push never silently proceeds.

set -Eeuo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: no 'origin' remote is configured. Add one before running this script." >&2
  exit 1
fi

# The paths below are intentionally explicit. This prevents accidentally committing
# an editor backup, credential file, or unrelated local work with this tutorial.
declare -a files=(
  ".env.example"
  ".github/workflows/ci.yml"
  ".gitignore"
  "Dockerfile"
  "README.md"
  "clip_projector/__init__.py"
  "clip_projector/config.py"
  "clip_projector/main.py"
  "clip_projector/model.py"
  "clip_projector/preprocessing.py"
  "clip_projector/schemas.py"
  "clip_projector/security.py"
  "clip_projector/service.py"
  "clip_projector/train.py"
  "docker-compose.yml"
  "docs/architecture.md"
  "docs/concepts.md"
  "docs/openapi.yaml"
  "docs/operations.md"
  "docs/security.md"
  "docs/serving.md"
  "docs/training.md"
  "k8s/clip-projector.yaml"
  "k8s/network-policy.yaml"
  "pyproject.toml"
  "scripts/load_test.js"
  "scripts/commit-each-file.sh"
  "tests/test_api.py"
  "tests/test_model.py"
  "tests/test_preprocessing.py"
  "tests/test_security.py"
  "tests/test_service.py"
)

# Confirm every changed/untracked path is in the allowlist before staging anything.
# --untracked-files=all is essential here: without it Git reports an untracked
# directory such as docs/ rather than each file inside it. The allowlist is made
# of files, so directory-level reporting would cause a safe but confusing abort.
# git status --porcelain is stable machine-readable output; the path begins at byte 4.
while IFS= read -r status_line; do
  [[ -z "$status_line" ]] && continue
  path="${status_line:3}"
  # A rename uses "old -> new" and is deliberately rejected rather than guessed.
  if [[ "$path" == *" -> "* ]]; then
    echo "ERROR: rename detected ($path); handle it manually before running this script." >&2
    exit 1
  fi
  known=false
  for allowed in "${files[@]}"; do
    [[ "$path" == "$allowed" ]] && known=true && break
  done
  if [[ "$known" == false ]]; then
    echo "ERROR: refusing to commit unexpected changed path: $path" >&2
    exit 1
  fi
done < <(git status --porcelain --untracked-files=all)

commit_one() {
  local file="$1"
  local message="$2"

  # A file may already be committed if this script is re-run after interruption.
  # Skip it only when it has no unstaged/staged/untracked change.
  if ! git status --porcelain -- "$file" | grep -q .; then
    echo "SKIP: $file has no pending change"
    return
  fi

  echo "STAGE:  $file"
  git add -- "$file"
  echo "COMMIT: $message"
  git commit -m "$message" -- "$file"
  echo "PUSH:   current branch to origin"
  git push origin HEAD
}

commit_one ".env.example" "chore: add example service configuration"
commit_one ".github/workflows/ci.yml" "ci: add quality and security pipeline"
commit_one ".gitignore" "chore: ignore local build artifacts"
commit_one "Dockerfile" "build: add hardened application container"
commit_one "README.md" "docs: add multimodal projector learning guide"
commit_one "clip_projector/__init__.py" "feat: initialize projector package"
commit_one "clip_projector/config.py" "feat: add validated service configuration"
commit_one "clip_projector/main.py" "feat: add authenticated inference API"
commit_one "clip_projector/model.py" "feat: implement CLIP dual encoder"
commit_one "clip_projector/preprocessing.py" "feat: add image and text preprocessing"
commit_one "clip_projector/schemas.py" "feat: define API request and response schemas"
commit_one "clip_projector/security.py" "feat: add API key and rate limit controls"
commit_one "clip_projector/service.py" "feat: add model inference service"
commit_one "clip_projector/train.py" "feat: add contrastive model training command"
commit_one "docker-compose.yml" "build: add local container orchestration"
commit_one "docs/architecture.md" "docs: explain system architecture and code flow"
commit_one "docs/concepts.md" "docs: explain CLIP concepts from first principles"
commit_one "docs/openapi.yaml" "docs: add OpenAPI service contract"
commit_one "docs/operations.md" "docs: add operations and recovery guide"
commit_one "docs/security.md" "docs: add security and privacy guide"
commit_one "docs/serving.md" "docs: explain API serving and deployment"
commit_one "docs/training.md" "docs: explain model training and evaluation"
commit_one "k8s/clip-projector.yaml" "deploy: add Kubernetes projector resources"
commit_one "k8s/network-policy.yaml" "deploy: deny projector egress by default"
commit_one "pyproject.toml" "build: add Python project metadata"
commit_one "scripts/load_test.js" "test: add API load test scenario"
commit_one "scripts/commit-each-file.sh" "chore: add one-file commit tutorial script"
commit_one "tests/test_api.py" "test: cover inference API behavior"
commit_one "tests/test_model.py" "test: cover CLIP model behavior"
commit_one "tests/test_preprocessing.py" "test: cover preprocessing validation"
commit_one "tests/test_security.py" "test: cover authentication and rate limiting"
commit_one "tests/test_service.py" "test: cover checkpoint loading failure"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: commits finished but the worktree is not clean:" >&2
  git status --short >&2
  exit 1
fi

echo "SUCCESS: every listed file was committed and pushed; git status is clean."
