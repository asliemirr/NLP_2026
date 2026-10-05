# Pipeline ve Inference — Hugging Face Transformers

Bu depo, Hugging Face `transformers` kütüphanesiyle doğal dil işleme (NLP) temellerini uygulamalı olarak gösteren tek hücreli bir Jupyter Notebook (`pipeline_ve_inference.ipynb`) içerir. Notebook üç ana konuyu ele alır: **pipeline'lar**, **transformer mimarileri** ve **metin üretimi (inference)**.

## İçerik

Notebook aşağıdaki bölümlerden oluşur:

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | Ortam ve Kurulum | Gerekli paketlerin kurulumu ve cihaz (CPU/GPU) tespiti |
| 2 | Pipeline'lar (Bölüm 1/3) | Duygu analizi, sıfır-örnek sınıflandırma, maskeli kelime tahmini |
| 3 | Transformer Mimarileri (Bölüm 1/5) | Encoder-only, Decoder-only ve Encoder-Decoder karşılaştırması |
| 4 | Metin Üretimi ve Inference (Bölüm 1/8) | GPT-2 ile farklı üretim stratejileri ve sıcaklık (temperature) analizi |

## Kullanılan Teknolojiler

- **Python**
- **PyTorch** — derin öğrenme altyapısı
- **Hugging Face Transformers** — önceden eğitilmiş modeller ve pipeline API'si
- **Matplotlib** — sıcaklık parametresinin olasılık dağılımına etkisinin görselleştirilmesi

## Kullanılan Modeller

- `distilbert-base-uncased-finetuned-sst-2-english` — duygu analizi
- `facebook/bart-large-mnli` — sıfır-örnek sınıflandırma
- `bert-base-uncased` — maskeli dil modeli (fill-mask)
- `gpt2` — metin üretimi

## Kurulum

Gerekli bağımlılıkları yükleyin:

```bash
pip install transformers torch accelerate matplotlib
```

## Kullanım

Notebook'u Jupyter, Google Colab veya VS Code üzerinden açıp tüm hücreleri sırayla çalıştırmanız yeterlidir:

```bash
jupyter notebook pipeline_ve_inference.ipynb
```

Not: Modeller ilk çalıştırmada Hugging Face Hub'dan indirilir, bu nedenle internet bağlantısı gerekir. Daha hızlı indirme ve yüksek hız limitleri için `HF_TOKEN` ortam değişkeni tanımlanabilir.

## Bölüm Detayları

### 1. Pipeline'lar

Hugging Face'in yüksek seviyeli `pipeline()` API'si ile üç farklı görev gösterilir:

- **Duygu analizi** (`sentiment-analysis`): Verilen bir cümlenin pozitif/negatif duygusunu ve güven skorunu döndürür.
- **Sıfır-örnek sınıflandırma** (`zero-shot-classification`): Modele önceden tanıtılmamış etiketler arasından en olası kategoriyi belirler.
- **Maskeli kelime tahmini** (`fill-mask`): Cümledeki `[MASK]` token'ı için en olası kelime önerilerini skorlarıyla listeler.

### 2. Transformer Mimarileri

Üç temel mimarinin karşılaştırıldığı bir tablo sunulur:

| Mimari Tipi | Temel Mekanizma | Örnek Modeller | Kullanım Alanları |
|-------------|-----------------|----------------|-------------------|
| Encoder-only | Çift yönlü (bi-directional) öz-dikkat | BERT, RoBERTa | Cümle sınıflandırma, NER, soru-cevap |
| Decoder-only | Tek yönlü (causal) maskeli öz-dikkat | GPT ailesi | Metin üretimi, yaratıcı yazım |
| Encoder-Decoder | Çapraz dikkat katmanlı Seq2Seq | T5, BART | Çeviri, özetleme, üretim |

### 3. Metin Üretimi ve Inference

GPT-2 modeli üzerinde beş farklı üretim stratejisi aynı girdi için denenir:

- **Greedy Search** — her adımda en yüksek olasılıklı token'ı seçer.
- **Beam Search** — birden fazla aday diziyi paralel tutarak daha iyi sonuç arar.
- **Düşük Sıcaklık (T=0.2)** — dağılımı keskinleştirir, daha belirleyici üretim.
- **Yüksek Sıcaklık (T=1.5)** — dağılımı yumuşatır, daha çeşitli/yaratıcı üretim.
- **Nucleus Sampling (p=0.9)** — kümülatif olasılığın %90'ını oluşturan token havuzundan örnekleme.

Ayrıca Matplotlib ile sıcaklık parametresinin softmax olasılık dağılımını nasıl değiştirdiği görselleştirilir.

## Çıktı Örneği

Duygu analizi örneği:

```
Sentiment Analysis Input: 'This Google Colab notebook for the NLP assignment is incredibly helpful and easy to follow!'
Output: [{'label': 'POSITIVE', 'score': 0.999...}]
```

## Lisans

Bu proje eğitim amaçlıdır. Kullanılan önceden eğitilmiş modellerin lisans koşulları Hugging Face Hub'daki ilgili model sayfalarında belirtilmiştir.
