# Lab 21 - Báo cáo đánh giá

**Học viên**: Mai Ngọc Duy - 2A202600736  
**Ngày nộp**: 2026-06-25  
**Hình thức nộp**: Option B - GitHub + HuggingFace Hub

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 mẫu
- **Chia dữ liệu**: 180 mẫu train / 20 mẫu eval, seed = 42
- **Định dạng dữ liệu**: Alpaca format, dùng các cột `instruction_vi`, `input_vi`, `output_vi`
- **Token length analysis**: min = 25, max = 738, p50 = 227, p95 = 562, p99 = 704
- **max_seq_length**: 1024, làm tròn từ p95 và giới hạn theo profile T4
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Phiên bản thư viện ghi nhận được**: Unsloth 2026.6.9, TRL 0.15.2, Transformers 5.5.0, PyTorch 2.11.0+cu128, bitsandbytes 0.49.2
- **Cấu hình train**: 3 epochs, batch size 1, gradient accumulation 8, effective batch size 8, learning rate 2e-4, cosine scheduler, optimizer `adamw_8bit`
- **Chi phí ước tính**: $0.08 cho 13.3 phút train, tính theo $0.35/giờ
- **HuggingFace Hub adapter**: [`Hugging Face`](https://huggingface.co/mzuyyy/qwen2.5-3b-vi-lab21-r16)

## 2. Kết quả thí nghiệm theo rank

| Rank | Alpha | Trainable Params | Thời gian train | Peak VRAM | Eval Loss | Perplexity |
|------|-------|------------------|-----------------|-----------|-----------|------------|
| 8 | 16 | 1,843,200 | 4.19 phút | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.54 phút | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 4.54 phút | 8.00 GB | 1.4768 | 4.3790 |

Ba adapter LoRA cho `r=8`, `r=16` và `r=64` đều đã được train và lưu lại. Notebook chưa tính perplexity của base model, nên báo cáo giữ mục này là "Chưa đo" thay vì tự ước lượng.

## 3. Phân tích loss curve

Trong notebook bản T4, eval-during-training được tắt để tiết kiệm VRAM, vì vậy loss curve chỉ có training loss của baseline `r=16`. Training loss nhìn chung giảm từ khoảng 1.61 ở step 5 xuống khoảng 1.39 ở step 65, dù vẫn có dao động nhỏ trong quá trình train.

![Loss curve](results/loss_curve.png "Loss curve")

Từ training loss đơn lẻ chưa thể kết luận chắc chắn có overfitting hay không, vì notebook không log eval loss ở từng epoch. Tuy nhiên, final eval perplexity của cả ba rank đều ở mức hợp lý và perplexity giảm dần khi rank tăng, cho thấy adapter đã học được tín hiệu từ dataset. Nếu muốn kiểm tra overfitting kỹ hơn, cần bật eval theo epoch hoặc dùng tập eval lớn hơn.

## 4. So sánh qualitative

Phần qualitative dùng 20 prompt đa dạng chủ đề. Notebook đã sinh output side-by-side cho 5 prompt đầu tiên, so sánh base model với adapter fine-tuned `r=16`.

### Ví dụ 1

**Prompt**: Viết email lịch sự xin đổi lịch họp với giảng viên vì trùng lịch phỏng vấn.

**Base**: Một email lịch sự xin đổi lịch họp với giảng viên vì trùng lịch phỏng vấn có thể như sau: "Tôi viết email này để xin phép phép thay đổi lịch họp của chúng ta..."

**Fine-tuned (r=16)**: Cảm ơn bạn đã thông báo cho tôi về lịch trình của bạn. Tôi rất tiếc vì lịch trình của chúng ta trùng với lịch trình của bạn...

**Nhận xét**: Kém hơn. Base model bám yêu cầu viết email tốt hơn, còn output fine-tuned giống phản hồi từ phía giảng viên hơn là email xin đổi lịch của sinh viên.

### Ví dụ 2

**Prompt**: Tóm tắt truyện Tấm Cám trong 5 câu, dùng giọng văn trung lập.

**Base**: Tấm Cám là câu chuyện cổ tích Trung Quốc cổ điển kể về một gia đình nghèo có ba người con gái...

**Fine-tuned (r=16)**: Tấm Cám là một câu chuyện cổ tích Trung Quốc cổ xưa. Câu chuyện kể về một chàng trai trẻ, một cô gái trẻ và một con cám...

**Nhận xét**: Kém hơn. Cả hai output đều có hallucination về nội dung truyện, nhưng fine-tuned model sai nhiều hơn và không đáp ứng rõ yêu cầu 5 câu.

### Ví dụ 3

**Prompt**: Giải thích lạm phát là gì cho học sinh lớp 8 bằng ví dụ đi chợ.

**Base**: Lạm phát là một tình trạng mà giá cả của các mặt hàng và dịch vụ tăng lên quá mức. Ví dụ, khi bạn đi chợ...

**Fine-tuned (r=16)**: Lạm phát là một tình trạng trong nền kinh tế, trong đó giá của các sản phẩm và dịch vụ tăng lên so với trước đó...

**Nhận xét**: Tốt hơn nhẹ. Output fine-tuned rõ ràng và tự nhiên hơn, nhưng vẫn nên có ví dụ đi chợ cụ thể hơn để phù hợp với học sinh lớp 8.

### Ví dụ 4

**Prompt**: Lập kế hoạch du lịch Đà Nẵng 3 ngày 2 đêm cho nhóm sinh viên, ngân sách vừa phải.

**Base**: Đà Nẵng là một điểm đến tuyệt vời cho nhóm sinh viên vì nó cung cấp nhiều hoạt động giải trí và trải nghiệm văn hóa...

**Fine-tuned (r=16)**: Lập kế hoạch du lịch Đà Nẵng 3 ngày 2 đêm cho nhóm sinh viên với ngân sách vừa phải, bạn có thể thực hiện theo các bước sau...

**Nhận xét**: Trung bình. Fine-tuned model có xu hướng trình bày theo dạng hướng dẫn, nhưng đoạn output được ghi lại vẫn khá chung chung.

### Ví dụ 5

**Prompt**: Viết đoạn quảng cáo ngắn cho một quán cà phê yên tĩnh gần trường đại học.

**Base**: Bạn có thể đã tìm kiếm một nơi yên tĩnh để học tập hoặc đọc sách ở trường đại học gần đó nhưng chưa tìm được...

**Fine-tuned (r=16)**: Bạn đang tìm kiếm một nơi để nghỉ ngơi và đọc sách gần trường đại học? Hãy đến quán cà phê yên tĩnh của chúng tôi...

**Nhận xét**: Tốt hơn. Output fine-tuned ngắn gọn, đúng giọng quảng cáo hơn và bám sát yêu cầu về quán cà phê yên tĩnh gần trường đại học.

## 5. Kết luận về trade-off giữa các rank

Trong thí nghiệm này, tăng rank giúp cải thiện eval perplexity theo thứ tự rõ ràng: `r=8` đạt 4.7479, `r=16` đạt 4.5544, và `r=64` đạt 4.3790. Điều này cho thấy rank cao hơn giúp adapter có nhiều năng lực biểu diễn hơn, từ đó học mapping instruction-response tốt hơn trên dataset tiếng Việt. Tuy nhiên, mức cải thiện từ `r=16` sang `r=64` chỉ khoảng 0.175 perplexity, trong khi số trainable parameters tăng từ 3.69M lên 14.75M, tức khoảng 4 lần. Training time trong notebook gần như không tăng vì dataset chỉ có 200 mẫu, nhưng peak VRAM của `r=64` là cao nhất, khoảng 8.00 GB. Nếu chỉ xét chất lượng định lượng, `r=64` là rank tốt nhất. Nếu xét ROI để chạy trên T4 hoặc triển khai thực tế, `r=16` là lựa chọn cân bằng hơn vì perplexity khá gần `r=64`, adapter nhỏ hơn nhiều và ít áp lực VRAM hơn. `r=8` phù hợp khi ưu tiên adapter nhỏ, nhưng chất lượng định lượng thấp nhất trong ba cấu hình.

## 6. Những điều học được

- Rank cao hơn không phải lúc nào cũng là lựa chọn tốt nhất cho production; cần cân bằng giữa perplexity, số tham số trainable, VRAM và kích thước adapter.
- Đánh giá qualitative là bắt buộc: dù perplexity tốt hơn, model fine-tuned vẫn có thể hallucinate hoặc trả lời kém hơn ở một số prompt ngoài phân phối dữ liệu train.
