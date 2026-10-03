# Bölüm 1: Transformer Modellerine Giriş

**Doğal Dil İşleme (NLP) Dersi · Dr. Mesut Polatgil · NLP_2026**

Bu klasör, [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) 1. bölümü (*Transformer Models*) için takımımızın Türkçe dokümantasyonunu, uygulama kodlarını, zafiyet testlerini ve sınav materyalini içerir.

## İçindekiler

1. [Takım ve Görev Dağılımı](#takim)
2. [Klasör Yapısı](#klasor-yapisi)
3. [Ortam Kurulumu ve Çalıştırma](#kurulum)
4. [Teorik Notlar ve Kod Açıklamaları](#unite-icerigi)
   - [1/1 · Giriş](#k1-1) — Özge Sayınbaş
   - [1/2 · Doğal Dil İşleme ve Büyük Dil Modelleri](#k1-2) — Özge Sayınbaş
   - [1/3 · Transformer'lar neler yapabilir?](#k1-3) — Himmet Can Umutlu
   - [1/4 · Transformer'lar nasıl çalışır?](#k1-4) — Simay Evin
   - [1/5 · 🤗 Transformer'lar görevleri nasıl çözer?](#k1-5) — Himmet Can Umutlu
   - [1/6 · Transformer Mimarileri](#k1-6) — Simay Evin
   - [1/8 · LLM'lerle çıkarım](#k1-8) — Himmet Can Umutlu
   - [1/9 · Önyargı ve sınırlamalar](#k1-9) — Abdulkadir Öcal
   - [1/10 · Ünite Özeti](#k1-10) — Özge Sayınbaş

---

<a id="takim"></a>

## 1. Takım ve Görev Dağılımı

| Üye | Rol | Zimmetli Konular | Çıktı |
|---|---|---|---|
| Özge Sayınbaş | Dokümantasyon Yöneticisi & Çeviri Sorumlusu (Öğrenci 1) | 1/1 Giriş, 1/2 NLP ve LLM'ler, 1/10 Ünite Özeti | [`turkce_ceviri.md`](turkce_ceviri.md), [`cheat_sheet.md`](cheat_sheet.md), `README.md` (birleştirme), PR |
| Himmet Can Umutlu | Repo Kaptanı & Uygulama Kodlama Mühendisi | 1/3 Transformer'lar Neler Yapabilir?, 1/5 Görev Çözümleri, 1/8 LLM ile Çıkarım | [`pipeline_ve_inference.ipynb`](pipeline_ve_inference.ipynb) |
| Simay Evin | Sistem Mimarı: Teori & Algoritma (Öğrenci 2) | 1/4 Transformer'lar Nasıl Çalışır?, 1/6 Transformer Mimarileri | `README.md` → Mimari ve Algoritma Analizi ([1/4](#k1-4), [1/6](#k1-6)) |
| Sıla Taşan | Sınav Komiseri & Ölçme Değerlendirme (Öğrenci 4) | 1/7 Hızlı Quiz, 1/11 Sertifikasyon Sınavı | `bolum_01_quiz_sinav.md` |
| Abdulkadir Öcal | QA / Red-Teamer (Öğrenci 3) | 1/9 Önyargı ve Sınırlamalar | [`red_teaming_zafiyet.ipynb`](red_teaming_zafiyet.ipynb) |

<a id="klasor-yapisi"></a>

## 2. Klasör Yapısı

```text
NLP_2026/
└── bolum_01_transformer_modellerine_giris/
    ├── README.md                    # Takımın ana dokümanı (birleştiren: Özge Sayınbaş)
    ├── turkce_ceviri.md             # Özge Sayınbaş: Bölüm 1'in bire bir Türkçe çevirisi
    ├── cheat_sheet.md               # Özge Sayınbaş: 1 sayfalık Transformer Cheat Sheet
    ├── pipeline_ve_inference.ipynb  # Himmet Can Umutlu: pipeline ve çıkarım parametreleri
    ├── red_teaming_zafiyet.ipynb    # Abdulkadir Öcal: halüsinasyon, önyargı ve zafiyet testleri
    └── bolum_01_quiz_sinav.md       # Sıla Taşan: quiz, sınav soruları ve çözüm anahtarı
```

Depo kuralları gereği bu klasörde yalnızca `.md`, `.qmd` ve `.ipynb` dosyaları bulunur.

<a id="kurulum"></a>

## 3. Ortam Kurulumu ve Çalıştırma

Kursun [kurulum bölümünde](https://huggingface.co/learn/llm-course/chapter0/1) belirtildiği gibi kütüphane şu komutla kurulur:

```bash
pip install "transformers[sentencepiece]"
```

Ders kuralı gereği notebook'lar PR öncesinde **Restart & Run All** ile baştan sona hatasız çalıştırılmalıdır.

---

<a id="unite-icerigi"></a>

## 4. Teorik Notlar ve Kod Açıklamaları (Kurs Sırasıyla)

Bu bölümde takım üyelerinin teorik notları ve kod açıklamaları, Hugging Face kursundaki alt bölüm sırasıyla yer alır. Her alt bölümün başında onu hazırlayan takım üyesinin adı yazılıdır. Bölümün Türkçe çevirisi [`turkce_ceviri.md`](turkce_ceviri.md), hızlı başvuru kartı [`cheat_sheet.md`](cheat_sheet.md) dosyasındadır.

<a id="k1-1"></a>

### 1/1 · Giriş

**Hazırlayan:** Özge Sayınbaş (Dokümantasyon Yöneticisi & Çeviri Sorumlusu) 

<a id="k1-2"></a>

### 1/2 · Doğal Dil İşleme ve Büyük Dil Modelleri

**Hazırlayan:** Özge Sayınbaş (Dokümantasyon Yöneticisi & Çeviri Sorumlusu)

<a id="k1-3"></a>

### 1/3 · Transformer'lar neler yapabilir?

**Hazırlayan:** Himmet Can Umutlu (Uygulama Kodlama Mühendisi) 

<!-- HIMMET CAN UMUTLU: Notebook'taki her bölüm için 1-2 cümlelik açıklamayı tabloya ekle. -->

| Notebook bölümü | Açıklama |
|---|---|
| Sentiment Analysis | <!-- açıklama --> |
| Zero-shot Classification | <!-- açıklama --> |
| Text Generation | <!-- açıklama --> |
| NER | <!-- açıklama --> |

<a id="mimari-analiz"></a>
<a id="k1-4"></a>

### 1/4 · Transformer'lar nasıl çalışır?

**Hazırlayan:** Simay Evin (Sistem Mimarı) · Mimari ve Algoritma Analizi

<!-- SIMAY EVIN: Alt başlıkların altına kendi metnini ekle. -->

#### Self-Attention ve Q, K, V matrisleri

<!-- Ders planına göre: Sorgu (Query), Anahtar (Key), Değer (Value) matrislerinin anlamı ve QK^T / sqrt(d_k) formülünün açıklaması -->

#### Transformer neden RNN/LSTM'den üstün? (Paralelleştirme)

<!-- Sıralı işleme vs. paralel işleme, uzun mesafe bağımlılıkları -->

#### Tahta şeması notları

<!-- Tahtada çizilecek blok şemasının adımları -->

<a id="k1-5"></a>

### 1/5 · 🤗 Transformer'lar görevleri nasıl çözer?

**Hazırlayan:** Himmet Can Umutlu (Uygulama Kodlama Mühendisi) 

<!-- HIMMET CAN UMUTLU: 1/5 ile ilgili notebook bölümlerini ve açıklamalarını buraya ekle. -->

<a id="k1-6"></a>

### 1/6 · Transformer Mimarileri

**Hazırlayan:** Simay Evin (Sistem Mimarı) · Mimari ve Algoritma Analizi

#### Mimari aileleri karşılaştırma tablosu

<!-- SIMAY EVIN: Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5) tablosu -->

<a id="k1-8"></a>

### 1/8 · LLM'lerle çıkarım

**Hazırlayan:** Himmet Can Umutlu (Uygulama Kodlama Mühendisi) 

| Notebook bölümü | Açıklama |
|---|---|
| Temperature, Top-k, Top-p | <!-- açıklama + grafik yorumu --> |
| Greedy Search vs. Beam Search | <!-- açıklama --> |

<a id="zafiyet-testleri"></a>
<a id="k1-9"></a>

### 1/9 · Önyargı ve sınırlamalar

**Hazırlayan:** Abdulkadir Öcal (Red-Teamer) 

<!-- ABDULKADİR ÖCAL: Her testin amacını ve bulgusunu aşağıdaki tabloya ekle. -->

| Test | Amaç | Bulgu |
|---|---|---|
| Mask filling cinsiyet/meslek önyargısı | <!-- --> | <!-- --> |
| Halüsinasyon testi | <!-- --> | <!-- --> |
| Prompt injection testi | <!-- --> | <!-- --> |

<a id="k1-10"></a>

### 1/10 · Ünite Özeti

**Hazırlayan:** Özge Sayınbaş (Dokümantasyon Yöneticisi & Çeviri Sorumlusu) 
