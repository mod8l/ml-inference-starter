# Load Test Report Template

Copy this template and fill it in after running `make load-test` or your own load test.

## Metadata

| Field | Value |
|---|---|
| Date | |
| Tested by | |
| Model version | |
| Image tag / commit | |
| Environment | local / staging / production |
| Load tool | locust / k6 / vegeta / other |

## Configuration

| Field | Value |
|---|---|
| Number of users | |
| Spawn rate | |
| Duration | |
| Payload size | |
| Batch size | |
| Instance type | |
| GPU / CPU | |
| Replicas | |

## Results

| Metric | Value |
|---|---|
| Total requests | |
| Requests per second | |
| p50 latency | |
| p95 latency | |
| p99 latency | |
| Error rate | |
| CPU utilization | |
| GPU utilization | |
| Memory utilization | |

## Observations

- What broke first?
- Where did latency start to degrade?
- Were there any errors or resource limits hit?

## Recommendations

- What should change before production?
- What should be load-tested next?
