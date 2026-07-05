# Cost Estimate

Rough cost guidance for running this reference architecture. Your actual costs depend on region, traffic, model size, and batching strategy.

## CPU inference

| Instance type | vCPU / RAM | Approx. cost / hour | Throughput (single req) | Cost / 1K requests |
|---|---|---|---|---|
| `c7i.large` | 2 / 4 GB | ~$0.05 | ~50 req/s | ~$0.0003 |
| `c7i.xlarge` | 4 / 8 GB | ~$0.10 | ~100 req/s | ~$0.0003 |

## GPU inference

| Instance type | GPU | Approx. cost / hour | Throughput (single req) | Cost / 1K requests |
|---|---|---|---|---|
| `g4dn.xlarge` | 1x T4 | ~$0.53 | ~500 req/s | ~$0.0003 |
| `g4dn.2xlarge` | 1x T4 | ~$0.75 | ~800 req/s | ~$0.0003 |

## Notes

- GPU instances are cost-effective only when utilization is high. At low utilization, CPU is usually cheaper.
- Batching increases throughput significantly; a GPU can often handle 10x more throughput with batching.
- Autoscaling adds cold-start latency. Set a minimum replica count if latency matters.
- Use spot/preemptible instances for dev/test and batch workloads.

## When to add a GPU

- Model inference takes > 100ms on CPU for your p99 latency target.
- You can keep GPU utilization above 40%.
- The cost of CPU scale-out exceeds the cost of a GPU instance at your traffic volume.
