# Operations, observability, and recovery guide

## Lifecycle: build to production

1. Run unit/API/security tests, static analysis, dependency audit, and container build in CI.
2. Train a candidate from a versioned manifest; generate an evaluation report and immutable signed checkpoint.
3. Create a release tuple: source commit, container digest, checkpoint digest, configuration version, evaluation report, and change approval.
4. Deploy to staging with production-like ingress, checkpoint mount, secret manager, and hardware. Run authenticated smoke and load tests.
5. Canary a small production fraction. Compare error rate, p95 latency, resource saturation, and sampled relevance to the prior version.
6. Progressively roll out only when gates pass. Keep the prior release tuple available for immediate rollback.

The Kubernetes deployment has three replicas as an availability baseline, health probes, HPA, and restrictive container settings. It is a template: production must supply the image registry, Secret, ingress, namespace policy, resource values measured for the actual model, and monitoring integration.

## Signals to collect

The application publishes `clip_projector_requests_total` by method/path/status and `clip_projector_request_duration_seconds` by path. Prometheus should scrape these from a private network. Gateway logs should add request ID, authenticated tenant/principal, response status, route, duration, bytes, and region; they must not include API keys, prompt/image bodies, raw embeddings, or query strings containing sensitive text. Add OpenTelemetry traces at the platform boundary if distributed request correlation is needed.

Use RED signals: **R**ate (requests), **E**rrors (5xx/429 and auth failures), and **D**uration (p50/p95/p99). Add resource signals: CPU/GPU utilization, memory working set, OOM kills, pod restarts, ready replicas, HPA desired/current count, queue/concurrency, checkpoint load failures, and ingress saturation. Model monitoring is separate: track candidate-vs-baseline retrieval quality, embedding norm/score distribution drift, domain/slice coverage, and feedback labels where responsibly available.

## SLOs and alerts

An initial availability objective can be 99.9% of valid authenticated requests returning non-5xx monthly. Set latency objectives only after target-hardware measurements; example starting goals are p95 text <300 ms and p95 image <1 s on a CPU baseline. Exclude client validation errors from availability but monitor spikes because they can reveal integration or abuse problems.

Page when 5xx exceeds 1% for 5 minutes, all ready replicas are unavailable, p95 breaches SLO for 10 minutes, OOM/restarts exceed two per pod/hour, checkpoint load fails, or authentication failures rise 5× a normal baseline. Ticket rather than page for sustained 429s, capacity >70%, dependency CVEs, or data/model drift unless customer impact reaches a defined threshold. Every alert needs an owner, dashboard link, severity, and runbook.

## Runbooks

### 5xx or unavailable service

Check ingress and `/healthz`, ready replica count, recent deployment/checkpoint/config change, pod events, restart/OOM status, and redacted error logs. If correlated to a release, shift traffic or `kubectl rollout undo deployment/clip-projector`, then verify authenticated text/image smoke tests. If no release changed, scale within safe capacity, inspect node/network conditions, and open an incident. Do not “fix” availability by disabling authentication, limits, or probes.

### Latency or saturation

Compare p95 by endpoint, request rate, image size distribution, CPU/GPU, memory, HPA behavior, and ingress queues. If demand is genuine, add measured capacity or GPU nodes; if abusive, enforce gateway quota/WAF. If a model change is correlated, roll back its release tuple. Confirm p95 after each action; horizontal scale cannot fix a single-pod memory leak or an oversized model beyond node capacity.

### Bad model quality

Stop rollout/promotion, return to the previous checkpoint, and preserve the failing release metadata. Compare preprocessing/tokenizer/model config, dataset digest, evaluation split, score distributions, and sampled failures. Do not tune thresholds against production anecdotes alone. Create a reproducible offline test case before retraining or changing prompts.

### Credential or artifact incident

Follow the actions in the security guide: revoke, rotate, isolate, preserve digests/evidence, restore known-good signed artifacts, then conduct a blameless post-incident review with corrective actions and owners.

## Backup and disaster recovery

There is no request database to restore. The recovery-critical assets are infrastructure manifests, secret-manager configuration, image registry, checkpoint/artifact registry, evaluation records, and training manifests. Replicate those across failure domains according to recovery objectives. Periodically perform a restore exercise: provision a clean namespace/cluster, restore secrets through approved workflow, deploy a prior signed release tuple, validate artifact checksum, confirm probes/metrics, then run authenticated smoke and retrieval tests. Record RTO/RPO from the exercise; do not claim a disaster-recovery target that has not been tested.
