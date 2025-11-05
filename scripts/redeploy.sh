#!/usr/bin/env bash
# Redeploy helper for Actuarial Chatbot (Docker-based)
# - Rebuild image, restart container, and verify health
# - Defaults align with README.MD deployment guidance

set -euo pipefail

# --------------- Logging ---------------
log_info()  { printf "\033[32m[INFO]\033[0m %s\n" "$1"; }
log_warn()  { printf "\033[33m[WARN]\033[0m %s\n" "$1"; }
log_error() { printf "\033[31m[ERROR]\033[0m %s\n" "$1"; }

# --------------- Defaults ---------------
IMAGE="${IMAGE:-actuarial-chatbot:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-actuarial-chatbot}"
ENV_FILE=""
USE_BUILDX="${USE_BUILDX:-false}"
PUSH="${PUSH:-false}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-}" # e.g., chatbotregistryaktuaria.azurecr.io/chatbot-app:latest
FLASK_PORT="${FLASK_PORT:-5000}"     # Container listens on 5000 per Dockerfile
VECTOR_BACKEND_DEFAULT="${VECTOR_BACKEND:-pinecone}"

# --------------- Helpers ---------------
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "Command '$1' not found. Please install it first."; exit 1;
  fi
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --env-file <path>    Path to env file (.env.production or .env)
  --image <name:tag>   Docker image tag (default: ${IMAGE})
  --container <name>   Container name (default: ${CONTAINER_NAME})
  --use-buildx         Use docker buildx for linux/amd64 builds
  --push               Push image to REGISTRY_IMAGE (requires REGISTRY_IMAGE)
  -h, --help           Show this help

Env vars respected:
  IMAGE, CONTAINER_NAME, USE_BUILDX=true|false, PUSH=true|false, REGISTRY_IMAGE
  FLASK_PORT (default 5000), VECTOR_BACKEND (default pinecone)
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env-file) ENV_FILE="$2"; shift 2;;
      --image) IMAGE="$2"; shift 2;;
      --container) CONTAINER_NAME="$2"; shift 2;;
      --use-buildx) USE_BUILDX="true"; shift;;
      --push) PUSH="true"; shift;;
      -h|--help) usage; exit 0;;
      *) log_error "Unknown option: $1"; usage; exit 1;;
    esac
  done
}

select_env_file() {
  if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    log_info "Using env file: $ENV_FILE"
    return
  fi
  if [[ -f .env.production ]]; then ENV_FILE=.env.production; log_info "Using .env.production"; return; fi
  if [[ -f .env ]]; then ENV_FILE=.env; log_info "Using .env"; return; fi
  log_warn "No env file found; proceeding without --env-file (not recommended)"
}

show_key_warnings() {
  # Basic validation for critical keys
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then log_warn "OPENAI_API_KEY not loaded; LLM features may fail"; fi
  if [[ -z "${PINECONE_API_KEY:-}" && "$VECTOR_BACKEND_DEFAULT" == "pinecone" ]]; then
    log_warn "PINECONE_API_KEY not loaded; Pinecone backend may fail"
  fi
}

build_image() {
  log_info "Building Docker image: $IMAGE"
  if [[ "$USE_BUILDX" == "true" ]]; then
    require_cmd docker
    docker buildx ls >/dev/null 2>&1 || { log_warn "buildx not configured; enabling default builder"; docker buildx create --use || true; }
    docker buildx build --platform linux/amd64 -t "$IMAGE" .
  else
    docker build -t "$IMAGE" .
  fi
  log_info "Image built: $IMAGE"
}

push_image() {
  if [[ "$PUSH" != "true" ]]; then return; fi
  if [[ -z "$REGISTRY_IMAGE" ]]; then log_error "REGISTRY_IMAGE not set for push"; exit 1; fi
  log_info "Tagging and pushing: $REGISTRY_IMAGE"
  docker tag "$IMAGE" "$REGISTRY_IMAGE"
  docker push "$REGISTRY_IMAGE"
  log_info "Pushed: $REGISTRY_IMAGE"
}

stop_container() {
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" || true
    log_info "Removing existing container: $CONTAINER_NAME"
    docker rm "$CONTAINER_NAME" || true
  fi
}

run_container() {
  log_info "Starting container: $CONTAINER_NAME"
  local run_args=(
    -d
    --name "$CONTAINER_NAME"
    --restart unless-stopped
    -p "${FLASK_PORT}:${FLASK_PORT}"
    -e FLASK_ENV=production
    -e FLASK_DEBUG=False
    -e FLASK_PORT="$FLASK_PORT"
    -e VECTOR_BACKEND="$VECTOR_BACKEND_DEFAULT"
  )
  if [[ -n "$ENV_FILE" ]]; then run_args+=( --env-file "$ENV_FILE" ); fi
  docker run "${run_args[@]}" "$IMAGE"
}

wait_health() {
  log_info "Waiting for health at http://localhost:${FLASK_PORT}/health"
  local tries=0 max=30 code=000
  until [[ "$code" == "200" || $tries -ge $max ]]; do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${FLASK_PORT}/health" || echo 000)
    tries=$((tries+1))
    log_info "Health check attempt ${tries}/${max}: HTTP ${code}"
  done
  if [[ "$code" != "200" ]]; then
    log_error "Service failed health check. Showing recent logs:"
    docker logs --since 5m "$CONTAINER_NAME" || true
    exit 1
  fi
  log_info "Service is healthy (HTTP 200)."
}

trap 'log_error "Redeploy failed."; echo "\nTips:"; echo "- Verify env keys OPENAI_API_KEY, PINECONE_API_KEY"; echo "- Check VECTOR_BACKEND and FLASK_PORT"; echo "- docker logs $CONTAINER_NAME"; exit 1' ERR

main() {
  require_cmd docker
  parse_args "$@"
  select_env_file

  # Load env into current shell if env file is present (for warnings)
  if [[ -n "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
  show_key_warnings

  build_image
  push_image
  stop_container
  run_container
  wait_health

  # Quick storage status
  log_info "Fetching storage status..."
  curl -s "http://localhost:${FLASK_PORT}/api/storage/status" | jq '.data.vector_store'
  log_info "Redeploy completed. Container: ${CONTAINER_NAME} Image: ${IMAGE}"
}

main "$@"