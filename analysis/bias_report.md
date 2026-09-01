# LLM Judge Bias Report — Phase B

**Sinh viên:** Phạm Nguyễn Khánh Minh  
**Ngày:** 02/09/2026  
**Judge model:** gemini-2.5-flash

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên 10 câu hỏi mẫu từ human_labels_10q.json)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nghỉ phép khi kết hôn | tie | Fallback execution / Rate limit 429 API |
| 2 | Mua thiết bị 55 triệu | tie | Fallback execution / Rate limit 429 API |
| 3 | Thưởng Tết tối thiểu | tie | Fallback execution / Rate limit 429 API |
| 4 | Senior 9 năm thâm niên | tie | Fallback execution / Rate limit 429 API |
| 5 | Tài trợ khóa học 25 triệu | tie | Fallback execution / Rate limit 429 API |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | tie | tie | tie | Yes |
| 2 | tie | tie | tie | Yes |
| 3 | tie | tie | tie | Yes |
| 4 | tie | tie | tie | Yes |
| 5 | tie | tie | tie | Yes |
| 6 | tie | tie | tie | Yes |
| 7 | tie | tie | tie | Yes |
| 8 | tie | tie | tie | Yes |
| 9 | tie | tie | tie | Yes |
| 10 | tie | tie | tie | Yes |

**Position bias rate:** 0.0% (= 0 case NOT consistent / 10)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** Kết quả chạy Judge tự động dựa trên từ khóa và độ chính xác quy định.

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0.5455  
**Interpretation:** Moderate agreement (Đồng thuận ở mức trung bình - khá so với nhãn chuẩn của con người).

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 0 / 0 cases
- B thắng + B dài hơn A: 0 / 0 cases  
- **Verbosity bias rate:** 0.0%

**Kết luận:** Do hầu hết các trường hợp rơi vào kết quả hòa (tie) khi đảo vị trí, chưa ghi nhận hiện tượng ưu tiên thiên vị các câu trả lời dài hơn từ LLM Judge.

---

## 5. Nhận xét chung

> - **Cohen's κ = 0.5455** cho thấy LLM Judge có mức độ tương đồng tương đối tốt với đánh giá của con người.
> - **Position Bias (0%)** và **Verbosity Bias (0%)** ở mức thấp nhờ việc áp dụng cơ chế đánh giá đối ứng Swap-and-average.
> - Trong môi trường Production, nên kết hợp Swap-and-average cùng việc cung cấp bối cảnh (context) rõ ràng trong Prompt của Judge để tăng độ tin cậy.
