# Model İçi İşleyiş ve Tokenizer — Hugging Face Transformers

Bu depo, Hugging Face `transformers` kütüphanesiyle modellerin perde arkasındaki işleyişini, tokenizer stratejilerini ve çoklu dizi yönetimini uygulamalı olarak gösteren tek hücreli bir Jupyter Notebook (`main.ipynb`) içerir. Notebook üç ana konuyu ele alır: **pipeline'ın iç yapısı (behind the pipeline)**, **tokenizer yaklaşımları (tokenizers)** ve **çoklu diziler ile attention mask mekanizması (handling multiple sequences)**.

## İçerik

Notebook aşağıdaki bölümlerden oluşur:

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | Ortam ve Kurulum | Gerekli paketlerin kurulumu ve donanım hızlandırıcı (CPU/GPU) tespiti |
| 2 | Pipeline'ın Arkasında Ne Var? (Bölüm 2/2) | Tokenizer -> Model -> Post-processing adımları ve Softmax dönüşümü |
| 3 | Tokenizer Stratejileri (Bölüm 2/4) | Word-based, Character-based, Subword-based karşılaştırması ve WordPiece demosu |
| 4 | Çoklu Diziler ve Attention Mask (Bölüm 2/5) | Padding boyutu eşitleme ve Attention Mask eksikliğinin logits üzerindeki bozucu etkisi |
| 5 | Görselleştirme | Attention Mask'in model çıktılarına etkisinin Matplotlib ile görselleştirilmesi |

## Kullanılan Teknolojiler

- **Python**
- **PyTorch** — tensör operasyonları ve derin öğrenme altyapısı
- **Hugging Face Transformers** — `AutoTokenizer` ve `AutoModelForSequenceClassification` sınıfları
- **NumPy & Matplotlib** — model logits sapmalarının ve dikkat maskesi etkisinin görselleştirilmesi

## Kullanılan Modeller

- `distilbert-base-uncased-finetuned-sst-2-english` — duygu analizi mimarisi üzerinden model içi tensör akışı ve logits analizi

## Kurulum

Gerekli bağımlılıkları yükleyin:

```bash
pip install transformers torch accelerate matplotlib numpy