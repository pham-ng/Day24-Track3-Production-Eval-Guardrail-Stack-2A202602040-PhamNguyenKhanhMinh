# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Phạm Nguyễn Khánh Minh  
**Ngày:** 01/09/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~10.29ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~2.54ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼ (~650ms P95)
[RAG Pipeline (Day 18 + Gemini-3.6-Flash)]
    │ M1 Chunk → M2 Search → M3 Rerank → Gemini-3.6-Flash
    ▼ (~2.50ms P95)
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Kết quả đo từ Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 8.48 | 10.29 | 10.29 | <10ms |
| NeMo Input Rail | 2.33 | 2.54 | 2.54 | <300ms |
| RAG Pipeline | 550.00 | 650.00 | 700.00 | <2000ms |
| NeMo Output Rail | 2.00 | 2.50 | 2.50 | <300ms |
| **Total Guard** | **10.85** | **12.71** | **12.71** | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Total guard latency chỉ mất 12.71ms P95 (thấp hơn nhiều so với ngân sách 500ms), do Presidio regex và NeMo rule-based checks vô cùng tối ưu và không tạo ra nghẽn cổ chai cho hệ thống.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | **0.8058** (Factual: 0.835, Multi-hop: 0.782, Adversarial: 0.801) |
| Worst metric | **context_recall** (Multi-hop: 0.578, Adversarial: 0.468) |
| Dominant failure distribution | **multi_hop** / **adversarial** |
| Cohen's κ | **0.5455** (Moderate agreement) |
| Adversarial pass rate | **16 / 20** (80.0%) |
| Guard P95 latency | **12.71 ms** |

---

## Nhận xét & Cải tiến

Hệ thống RAG Eval & Guardrail Stack đã hoàn thành tốt các yêu cầu với 40/40 test cases passed.
1. **Điểm mạnh:** Lớp Presidio PII và NeMo Input Guardrail phản ứng rất nhanh (P95 latency 12.71ms), chặn được 80% (16/20) các cuộc tấn công jailbreak, prompt injection và PII leak.
2. **Điểm cần cải thiện:** `context_recall` ở tập `multi_hop` (0.578) và `adversarial` (0.468) còn thấp do thiếu các chunk tổng hợp và việc bỏ qua 2 file PDF scan (chưa OCR).
3. **Đề xuất Production:** Cần tích hợp Tesseract/Unstructured OCR để index toàn bộ văn bản scanned PDF, bổ sung BM25 Hybrid Search và fine-tune Colang rules cho NeMo Guardrails để đạt block rate >95%.
