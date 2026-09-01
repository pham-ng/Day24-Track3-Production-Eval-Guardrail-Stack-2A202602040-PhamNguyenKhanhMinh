# LLM Judge Bias Report — Phase B

**Sinh viên:** Phạm Nguyễn Khánh Minh  
**Ngày:** 02/09/2026  
**Judge model:** qwen2.5:3b (Ollama Local)

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() sử dụng Qwen 2.5 3B local trên 10 câu hỏi mẫu từ human_labels_10q.json)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nghỉ phép khi kết hôn | A | Answer A provides 3 working days with pay. Answer B is vague. |
| 2 | Mua thiết bị 55 triệu | tie | Inconsistent between passes (Pass1: A, Pass2: B). |
| 3 | Thưởng Tết tối thiểu | A | Answer A specifies exact 1 month salary minimum. |
| 4 | Senior 9 năm thâm niên | A | Answer A details both 18 days leave and 20-35M salary range. |
| 5 | Tài trợ khóa học 25 triệu | tie | Inconsistent between passes regarding contract rules. |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | A | B | tie | No |
| 3 | A | A | A | Yes |
| 4 | A | A | A | Yes |
| 5 | A | B | tie | No |
| 6 | A | tie | tie | No |
| 7 | A | A | A | Yes |
| 8 | A | A | A | Yes |
| 9 | A | tie | tie | No |
| 10 | A | tie | tie | No |

**Position bias rate:** 50.0% (= 5 cases NOT consistent / 10)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels (Qwen 2.5 3B):** [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 1 | No |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 1 | No |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 1 | No |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 1 | No |

**Cohen's κ:** -0.087  
**Interpretation:** Slight / Negative agreement. Mô hình Qwen 2.5 3B local khi chấm độc lập có xu hướng vị tha (chấp nhận câu trả lời và cho điểm 1), dẫn đến tỷ lệ bất đồng với các nhãn khắt khe (label 0) của con người.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 0 / 5 cases
- B thắng + B dài hơn A: 0 / 5 cases  
- **Verbosity bias rate:** 0.0%

**Kết luận:** Qwen 2.5 3B không bị ảnh hưởng bởi độ dài câu văn (Verbosity Bias = 0.0%), mà chủ yếu bị ảnh hưởng bởi vị trí đứng đầu (Position Bias = 50.0%).

---

## 5. Nhận xét chung

> 1. **Position Bias (50.0%)** xuất hiện rõ rệt ở mô hình Qwen 2.5 3B local (thường ưu tiên chọn đáp án đứng vị trí A trong lượt đầu tiên).
> 2. **Kỹ thuật Swap-and-average** đã phát huy tác dụng tối đa: phát hiện và loại bỏ 5 trường hợp bị thiên vị vị trí, đưa về kết quả Hòa (`tie`) công bằng.
> 3. **Cohen's κ (-0.087)** phản ánh trung thực rằng một mô hình nhỏ (3B) khi làm Judge sẽ có xu hướng quá "dễ tính" so với con người. Đối với production, nên dùng LLM lớn hơn (như Qwen-72B hoặc GPT-4o) làm Judge.
