# Training, data, evaluation, and model promotion

## Before writing code: define the task

Decide what “good” means. Product-image search, medical-image retrieval, and meme-caption matching have different data, risk, and evaluation needs. Write a data card that records data source, license, consent/legal basis, geography/language coverage, known exclusions, labeling rules, retention period, and contact owner. Do not scrape or train on material simply because it is technically available.

## Dataset format and splits

The training CLI consumes JSONL, one object per line:

```json
{"image":"animals/dog-001.jpg","text":"a brown dog running on grass"}
```

`image` is a path relative to `--image-root`; `text` is its positive caption. Validate every path stays beneath the configured root before accepting an untrusted manifest in a shared environment. This implementation expects the training process and manifest owner to be trusted; production pipelines should add path containment checks, malware scanning, content policy filtering, image dimension limits, duplicate detection, and provenance checks.

Split data by source/entity rather than random file when near-duplicates or the same product/person can occur in several images. Otherwise test retrieval can look excellent because training has already seen effectively the same sample. Keep a locked test set untouched until model comparison. Version the exact manifest, not just a folder name.

## Training loop explained

Run:

```bash
python -m clip_projector.train \
  --manifest data/train.jsonl --image-root data/images \
  --output checkpoints/clip-projector-v1.pt --epochs 5 --batch-size 32 \
  --learning-rate 0.0005 --device cuda
```

For every batch, `DataLoader` obtains image tensors and captions. `batch_tokens` creates a fixed `[batch, context_length]` tensor. `model.contrastive_loss` forms all pairwise scores and cross entropy targets `[0, 1, ..., batch-1]`. `backward()` computes gradients; gradient clipping limits their global norm; `AdamW.step()` changes weights. The checkpoint stores `state_dict` under `model`.

The command is intentionally minimal, not an experiment platform. A serious training run also records random seeds, deterministic settings, train/validation loss per epoch, LR schedule, mixed precision policy, distributed topology, data failures, gradient norms, and periodic checkpoints. It validates on a held-out set every epoch and retains the best checkpoint by pre-declared metric—not by a post-hoc choice.

## Hyperparameters that matter

Batch size defines the number of in-batch negatives. Larger batches usually improve contrastive learning but require more memory and can need learning-rate scaling. Learning rate too high diverges; too low learns slowly. Weight decay limits parameter growth. Image size and patch size set visual detail and token count: smaller patches see finer detail but attention costs grow roughly quadratically in number of tokens. Context length truncates text. Embedding dimension controls the shared-space capacity and vector storage cost. Tune one controlled change at a time and compare against a fixed baseline.

## Evaluation: loss is not enough

Contrastive loss measures the training objective, not product utility. For retrieval, compute a score matrix on held-out pairs. `Recall@K` is the fraction of queries whose correct item appears in the top K. Report image-to-text and text-to-image separately, plus median rank and confidence intervals. For zero-shot classification, encode prompt templates for each label, choose the highest similarity, and report class-balanced precision/recall/F1—not just accuracy. Slice every metric by language, source, image quality, demographic group where lawful/appropriate, and safety category.

Perform qualitative error review: are failures caused by captions, visual ambiguity, culture/language, OCR, rare concepts, or a shortcut such as background colour? Test known policy-sensitive and adversarial cases. Measure serving p50/p95 latency and memory using the exact candidate checkpoint and target hardware. A model should pass predefined quality, fairness/safety, robustness, and performance gates before promotion.

## Promotion and rollback

Create a release record containing the dataset and checkpoint SHA-256, model/tokenizer config, evaluation report, code/container digest, owner, date, and approval. Upload immutable artifact(s), deploy first to a canary, compare online technical metrics and sampled relevance against the prior release, then progressively roll out. Rollback means re-deploying the entire prior release record. Never overwrite a checkpoint path in place; that destroys auditability and makes failures irreproducible.

## Common failure modes

Collapse (all embeddings similar) can result from optimization/configuration errors; inspect embedding norms, score distribution, loss, and gradients. Memorization/overfitting shows a train/validation gap; improve splits/data diversity or regularization. Bad captions create contradictory positives; clean data. Low recall on a domain often means coverage is missing, not that another epoch will help. Token hash collisions and English-centric preprocessing limit language quality—use a tested versioned tokenizer before scaling.
