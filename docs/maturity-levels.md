# Maturity Levels: From Prototype to Production

This repo is intentionally small, but the same architecture can grow with your team. I use these maturity levels to decide which investments come next.

## Level 0 — Notebook prototype

- Model runs in a notebook or local script.
- No serving layer, no version control for data.
- **Goal:** prove the problem is solvable.

## Level 1 — Local API

- Model is wrapped in a local HTTP API (`make dev`).
- Input/output contracts are defined.
- Basic unit tests exist.
- **Goal:** validate the integration shape.

## Level 2 — Containerized service

- Service runs in Docker (`make docker-build`).
- Health and readiness probes exist.
- Prometheus metrics are exposed.
- **Goal:** run on a single node with realistic load.

## Level 3 — Kubernetes deployment

- Manifests applied to a cluster (`kubectl apply -f k8s/`).
- HPA and resource limits configured.
- ServiceMonitor or equivalent scrape config in place.
- **Goal:** survive node failures and scale with traffic.

## Level 4 — Production operation

- Full observability stack (Prometheus + Grafana).
- Model registry and canary deployments.
- Cost per prediction is tracked.
- Incident runbooks and rollback playbooks exist.
- **Goal:** operate as a business-critical service.

## When to move between levels

Move up a level when the previous one is **boring** — meaning it runs for a week without surprises. Skipping levels to impress investors or meet a deadline usually creates debt that shows up at 3 a.m.
