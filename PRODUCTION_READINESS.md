# Production Readiness Checklist

This checklist maps the `ml-inference-starter` architecture to the practices I expect before a model serving system sees customer traffic. Use it as a template for your own service.

## Operational readiness

| Item | Status | Owner | Evidence |
|------|--------|-------|----------|
| Health (`/health`) and readiness (`/ready`) probes are configured and tested | | | |
| Graceful startup/shutdown behavior is documented | | | |
| Rolling updates and rollback procedure are tested | | | |
| Resource requests and limits are set (CPU, memory, GPU) | | | |
| Autoscaling policy is configured and load-tested | | | |
| Runbook exists for the three most likely failure modes | | | |

## Observability

| Item | Status | Owner | Evidence |
|------|--------|-------|----------|
| Request count, latency, and error rate metrics are exposed | | | |
| Latency is broken down by percentile (p50, p95, p99) | | | |
| Model version and prediction distribution are logged per request | | | |
| Alerts exist for high error rate, high latency, and low throughput | | | |
| Dashboard is reviewed by on-call engineers | | | |

## Model lifecycle

| Item | Status | Owner | Evidence |
|------|--------|-------|----------|
| Model artifact is versioned and immutable | | | |
| Training-serving skew is measured and bounded | | | |
| Shadow or canary deployment process is defined | | | |
| Rollback to previous model version is < 5 minutes | | | |
| Model card or equivalent documentation exists | | | |

## Security and reliability

| Item | Status | Owner | Evidence |
|------|--------|-------|----------|
| Input validation rejects malformed or oversized payloads | | | |
| Secrets are not baked into the container image | | | |
| Network policies restrict egress to required endpoints | | | |
| Dependency scanning is part of CI | | | |
| Load test results are documented (see [`docs/load-test-report-template.md`](docs/load-test-report-template.md)) | | | |

## Cost and efficiency

| Item | Status | Owner | Evidence |
|------|--------|-------|----------|
| Cost per 1K predictions is estimated (see [`docs/cost-estimate.md`](docs/cost-estimate.md)) | | | |
| GPU utilization target is defined | | | |
| Scaling to zero (or near-zero) is considered for dev/test | | | |
| Right-sizing is revisited after 30 days of production traffic | | | |
