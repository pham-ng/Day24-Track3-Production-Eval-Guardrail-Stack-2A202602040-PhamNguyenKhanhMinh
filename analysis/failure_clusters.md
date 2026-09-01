# Failure Cluster Analysis — Phase A

**Sinh viên:** Phạm Nguyễn Khánh Minh  
**Ngày:** 02/09/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 1.000 | 1.000 | 1.000 |
| answer_relevancy | 0.488 | 0.243 | 0.278 |
| context_precision | 0.950 | 0.900 | 1.000 |
| context_recall | 0.694 | 0.400 | 0.363 |
| **avg_score** | **0.783** | **0.636** | **0.660** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Nhân viên tự ý xóa malware khỏi máy và chia sẻ thông tin sự cố lên mạng xã hội bị xử lý kỷ luật thế nào? | 0.3850 | answer_relevancy |
| 2 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm bảo mật dữ liệu khách hàng. Quy trình báo cáo và mức thưởng phát hiện vi phạm? | 0.4230 | answer_relevancy |
| 3 | multi_hop | Nhân viên đi công tác nước ngoài 4 ngày tại thành phố nhóm 1: tổng phụ cấp công tác phí và mức tạm ứng tối đa? | 0.5240 | answer_relevancy |
| 4 | multi_hop | Nhân viên Junior P1 có lương cơ bản 12 triệu vừa hoàn thành chứng chỉ AWS Pro. Tổng thu nhập tháng tiếp theo là bao nhiêu? | 0.5430 | answer_relevancy |
| 5 | factual | Cơ cấu điểm đánh giá hiệu suất gồm những thành phần nào và trọng số ra sao? | 0.5710 | answer_relevancy |
| 6 | adversarial | Theo chính sách nghỉ phép cũ (v2023), nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.6060 | answer_relevancy |
| 7 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.6060 | answer_relevancy |
| 8 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu và thời gian thay đổi bắt buộc? | 0.6110 | answer_relevancy |
| 9 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm? | 0.6200 | answer_relevancy |
| 10 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép năm? | 0.6210 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |
| context_precision | 1 | 0 | 0 | 1 |
| answer_relevancy | 19 | 20 | 10 | 49 |
| **Total** | **20** | **20** | **10** | **50** |

---

## 4. Root Cause & Insight Analysis

- **Dominant Failure Distribution:** `multi_hop` & `adversarial`
- **Dominant Failure Metric:** `answer_relevancy`
- **Insight Báo cáo trung thực (Qwen 2.5 3B Real Execution):**
  - Mô hình `qwen2.5:3b` suy luận chính xác và trung thực với context (`faithfulness` đạt **1.0**).
  - Điểm `answer_relevancy` thấp ở tập `multi_hop` (**0.243**) và `adversarial` (**0.278**) do câu hỏi đòi hỏi tổng hợp từ nhiều nguồn tài liệu (tính toán phụ cấp, thưởng, công tác phí) và câu hỏi bẫy bối cảnh cũ/mới.
  - Điểm `context_recall` ở tập `adversarial` (**0.363**) bị sụt giảm do thiếu ngữ cảnh từ 2 file PDF dạng scan (chưa qua OCR).
- **Giải pháp nâng cấp:**
  1. Thêm Tesseract/Unstructured OCR để index tài liệu PDF scan.
  2. Bổ sung Prompt Template dạng CoT (Chain-of-Thought) cho Qwen để hỗ trợ suy luận đa bước ở nhóm `multi_hop`.
