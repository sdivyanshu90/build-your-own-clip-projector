# Security, privacy, and abuse-resistance guide

Security begins by identifying assets: API credentials; user images, text, and embeddings; model weights/checkpoints; training data; deployment credentials; and service availability. An embedding can be sensitive because it may support inference about its source or be linked with other data. Protect it according to the classification of the underlying content.

## Threat model

| Actor / attack | Consequence | Controls in this project | Required platform control |
|---|---|---|---|
| Anonymous caller sends huge/malformed images | Memory/CPU exhaustion or decoder exploit | Byte cap, MIME allowlist, decoder errors handled | WAF body limit, concurrency/timeout, patched base image |
| Stolen API key | Unauthorized inference/data disclosure | Constant-time key check, no key logging | Secret manager, rotation, tenant identity, anomaly detection |
| Caller floods many pods | Cost/availability loss | Per-pod sliding window | Gateway/global quota and DDoS protection |
| Web caller abuses CORS | Cross-origin data use | Explicit allowlist, no cookie auth | Backend-for-frontend and CSP for any UI |
| Attacker supplies model artifact | Remote code/data compromise or poisoned model | `weights_only=True`, explicit path | Signed immutable registry, checksum/approval, write separation |
| Developer logs payloads | Privacy breach | Application does not log bodies | Log redaction, access control, retention policy |
| Compromised pod calls internet | Exfiltration/download | No runtime HTTP feature | Egress-deny NetworkPolicy, workload identity |
| Training data is poisoned/biased | Unsafe or degraded model | Evaluation guidance | Data provenance, review, red-team and promotion gates |

## Request security controls in detail

Authentication uses `X-API-Key`, appropriate for service-to-service callers. Keys must be randomly generated (at least 32 bytes), stored only in a secret manager, injected at runtime, rotated with an overlap window, and scoped per environment/tenant. This repository compares key candidates using `hmac.compare_digest`, which avoids a simple timing leak. The service should not become the primary identity provider: an API gateway should validate OIDC/JWT or mTLS, apply authorization policy, add a request ID, and pass only trusted identity context.

Authorization answers “what is this caller allowed to do?”, which API key validity alone does not express. For a multi-tenant product, enforce tenant/model/region permissions at gateway and attach tenant identity to a downstream vector store filter. Never accept a tenant ID from an unverified body/header as authorization evidence.

Input validation is layered. HTTP schema validation bounds text count and length. Upload reading bounds bytes. MIME allowlisting reduces decoder surface. Pillow must decode the actual bytes. Tokenization bounds context. Tensor shapes are fixed. No SQL, shell, URL fetching, template rendering, or deserialization of request data occurs; this prevents the common injection/SSRF paths by design. Continue to patch Pillow, PyTorch, FastAPI, and the base image because input validation does not erase implementation vulnerabilities.

## Checkpoint and supply-chain security

PyTorch checkpoints are not inherently safe interchange formats: traditional pickle-based loading can execute code. Load only approved artifacts; `weights_only=True` narrows risk but does not make unknown artifacts trustworthy. Build checkpoints in a controlled CI/training account, compute SHA-256, sign the artifact/release metadata, store it immutably, and grant serving identity read-only access to a specific digest. Separate the people/systems that can train, approve, and deploy. Pin container base images by digest in production; generate an SBOM; scan dependencies and images; fail release on policy-defined critical vulnerabilities.

## OWASP mapping

| OWASP Top 10 | Relevant response |
|---|---|
| A01 Broken Access Control | Authenticate inference; gateway tenant authorization; private metrics; least-privilege Kubernetes RBAC. |
| A02 Cryptographic Failures | TLS 1.2+ at ingress; encrypted volumes/object storage; secret manager; no payloads in logs. |
| A03 Injection | Typed JSON/multipart, bounded tensors, no database/shell/URL input path. |
| A04 Insecure Design | Explicit stateless boundary, threat model, quotas, safety and promotion gates. |
| A05 Security Misconfiguration | Non-root/read-only container, no capabilities, explicit CORS, production config review. |
| A06 Vulnerable Components | CI static/security scans, dependency patch SLA, signed images. |
| A07 Identification/Authentication Failures | High-entropy rotating keys; gateway OIDC/mTLS; anomaly alerts. |
| A08 Software/Data Integrity Failures | Signed checkpoint/container, immutable release record, protected CI. |
| A09 Logging/Monitoring Failures | Request/status/latency metrics, redacted structured logs, alerting and incident process. |
| A10 SSRF | No URL ingestion or runtime outbound fetch; deny egress. |

## Privacy and AI-specific risks

Do not assume “we only store embeddings” resolves privacy. Avoid retaining request data by default; when product requirements require it, document purpose, legal basis, regional location, access control, deletion mechanism, retention duration, and incident notification obligations. Do not use production requests for training without a documented consent/notice and review. Evaluate model behavior on protected and vulnerable groups where appropriate and lawful. CLIP-like similarity can generate harmful, biased, or misleading associations; do not use it as the sole basis for high-impact decisions, identity claims, medical/legal conclusions, or automated enforcement without domain-specific governance and human review.

## Incident actions

For a credential leak, immediately revoke/rotate at the secret manager and gateway, deploy replacement secrets, determine blast radius from access metadata, and preserve evidence without copying payloads. For decoder/model compromise, isolate the namespace, stop traffic, record image/checkpoint/container digests, rebuild from patched signed artifacts, and restore the last known-good release. For suspected model-quality harm, pause promotion, roll back, preserve evaluation data and decisions, and investigate dataset/model/version changes through the release record.
