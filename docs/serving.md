# Serving, API, and deployment guide

## What the service guarantees

The service is an inference boundary. It receives bounded, authenticated input, returns vectors or one cosine score, exposes readiness/metrics, and holds no request database. It does not train, fetch remote images, make content-policy decisions, guarantee semantic correctness, or provide vector search. These non-goals are security and reliability boundaries, not missing conveniences.

## Endpoints and examples

`GET /healthz` answers `{"status":"ok","model_loaded":true}` and is used by probes. `GET /metrics` emits Prometheus format and should be limited to the monitoring network.

`POST /v1/embeddings/text` accepts JSON:

```json
{"texts":["a dog on a beach", "a red sports car"]}
```

It returns the same number of float vectors and their dimension. `POST /v1/embeddings/image` accepts multipart form field `image`. `POST /v1/similarity?text=a%20dog` accepts that same image and returns one cosine similarity. Send `X-API-Key` on all inference calls. The authoritative machine contract is [openapi.yaml](openapi.yaml).

HTTP responses are intentional: 401 means a missing/invalid key; 413 means empty or too large body; 415 means a declared image type outside JPEG/PNG/WebP; 422 means schema/blank text/decoder failure; 429 means the local rate window is exhausted; 500 indicates an unexpected server failure. Clients should not retry validation failures. Retry 429 only with exponential backoff and a product-level budget. Retry 5xx only for idempotent requests and with jitter.

## Environment settings

All settings use `CLIP_` prefix. `CLIP_API_KEYS` is a comma-separated rotation set, `CLIP_REQUIRE_API_KEY` defaults true, `CLIP_MAX_UPLOAD_BYTES` defaults 5 MiB, `CLIP_RATE_LIMIT_PER_MINUTE` defaults 60 per source IP per pod, `CLIP_MODEL_DEVICE` is normally `cpu` or `cuda`, and `CLIP_CHECKPOINT_PATH` points to a read-only evaluated checkpoint. `CLIP_CORS_ORIGINS` is an explicit comma-separated allowlist. Keep CORS empty unless a browser origin truly needs access; never combine permissive origins with credentialed browser sessions.

## Deployment topology

Terminate TLS, request IDs, WAF rules, global quotas, and enterprise identity at ingress/API gateway. The Kubernetes manifest runs three non-root replicas with read-only root filesystem, dropped capabilities, probes, CPU/memory limits, and HPA. Mount a checkpoint and secrets read-only. Apply default-deny egress unless operationally required. Use a GPU node pool only after measuring its throughput/cost; CPU replicas may be preferable at modest volume.

At minimum, set an ingress body-size limit consistent with the API, configure request timeout above measured p99 plus margin, and ensure load balancer health probes do not require API keys. Keep `/metrics` private. Do not expose a development `.env`, model files, or debug stack traces publicly.

## Capacity planning

Measure with the actual image distribution, checkpoint, batch/concurrency, and hardware. Image decode plus Vision Transformer normally costs more than text encoding. A single request’s memory is model weights plus activation/decode buffers; body limit alone is not a concurrency limit. Set gateway concurrency/queue controls, resource limits, and HPA targets from load testing. The provided k6 test is a starting smoke workload, not evidence of production capacity. Test overload, restart during traffic, bad images, checkpoint-mount loss, and one-zone failure.

## Integration patterns

For search, embed documents/images asynchronously, store vectors in a dedicated vector index with metadata and ACL filters, then embed the query and run nearest-neighbour retrieval. Store model version with every vector; indexes must support re-embedding during model changes. For ranking, use cosine similarity as one feature, not necessarily the only decision. For zero-shot labels, encode several carefully chosen prompt templates per label and compare against the image; validate the template choice on held-out data.
