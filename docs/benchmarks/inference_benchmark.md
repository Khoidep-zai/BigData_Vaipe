# Inference Benchmark (Web Backend)

Source: docs/benchmarks/inference_benchmark.json
Method: local Flask test client, includes preprocessing + request handling overhead.

## Hardware Mode Summary

| Device Mode | Availability | Note |
|---|---|---|
| CPU | Available | Measured successfully |
| CUDA | Not available | CUDA not available in this environment |

## Latency Results (ms)

| Endpoint Scenario | Device | Runs | Mean | P50 | P95 | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| classify_single_image | cpu | 12 | 524.34 | 523.44 | 572.28 | 492.86 | 608.54 |
| check_prescription_two_pills | cpu | 8 | 2062.26 | 2038.40 | 2348.78 | 1764.43 | 2408.32 |
| classify_single_image | cuda | - | N/A | N/A | N/A | N/A | N/A |
| check_prescription_two_pills | cuda | - | N/A | N/A | N/A | N/A | N/A |

## Interpretation

- Endpoint classify_single_image trung bình khoảng 0.52 giây trên CPU.
- Endpoint check_prescription_two_pills trung bình khoảng 2.06 giây trên CPU.
- check_prescription chậm hơn do batch inference nhiều ảnh và bước đối chiếu ngữ cảnh toa thuốc.
