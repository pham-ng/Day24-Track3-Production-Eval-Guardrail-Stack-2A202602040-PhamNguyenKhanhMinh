# Failure Cluster Analysis — Phase A

**Sinh viên:** Phạm Nguyễn Khánh Minh  
**Ngày:** 02/09/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.000 | 0.000 | 0.000 |
| answer_relevancy | 0.000 | 0.000 | 0.000 |
| context_precision | 0.000 | 0.000 | 0.000 |
| context_recall | 0.000 | 0.000 | 0.000 |
| **avg_score** | **0.000** | **0.000** | **0.000** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | factual | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn? | 0.0000 | faithfulness |
| 2 | factual | Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên? | 0.0000 | faithfulness |
| 3 | factual | Phụ cấp ăn trưa hàng tháng là bao nhiêu? | 0.0000 | faithfulness |
| 4 | factual | Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không? | 0.0000 | faithfulness |
| 5 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.0000 | faithfulness |
| 6 | factual | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | 0.0000 | faithfulness |
| 7 | factual | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.0000 | faithfulness |
| 8 | factual | Nhân viên được nghỉ bao nhiêu ngày khi cha hoặc mẹ mất? | 0.0000 | faithfulness |
| 9 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.0000 | faithfulness |
| 10 | factual | Nhân viên chính thức được phép làm việc từ xa tối đa bao nhiêu ngày một tuần? | 0.0000 | faithfulness |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 20 | 20 | 10 | 50 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |
| **Total** | **20** | **20** | **10** | **50** |

---

## 4. Root Cause & Insight Analysis

- **Dominant Failure Distribution:** `factual`
- **Dominant Failure Metric:** `faithfulness`
- **Insight Báo cáo trung thực:** Trong quá trình chạy đánh giá RAGAS với mô hình `gemini-2.5-flash`, hệ thống đã gặp phải giới hạn tần suất gọi API (Rate Limit 429 - Exceeded free tier quota). Kết quả dẫn đến việc câu trả lời chưa trích xuất được đủ thông tin chính xác từ ngữ cảnh.
- **Giải pháp đề xuất:** Siết chặt system prompt, tăng thời gian backoff/retry giữa các API calls hoặc chuyển sang sử dụng API Key Tier cao hơn để thu được điểm số đánh giá phản ánh chính xác hiệu năng RAG.
