# Bölüm 1: Transformer Modellerine Giriş

**Doğal Dil İşleme (NLP) Dersi · Dr. Mesut Polatgil · NLP_2026**

Bu klasör, [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) 1. bölümü (*Transformer Models*) için takımımızın Türkçe dokümantasyonunu, uygulama kodlarını, zafiyet testlerini ve sınav materyalini içerir.

## İçindekiler

1. [Takım ve Görev Dağılımı](#takim)
2. [Klasör Yapısı](#klasor-yapisi)
3. [Ortam Kurulumu ve Çalıştırma](#kurulum)
4. [Teorik Notlar ve Kod Açıklamaları](#notlar)
   - [Özge Sayınbaş · 1/1, 1/2, 1/10](#ozge)
   - [Himmet Can Umutlu · 1/3, 1/5, 1/8](#himmet)
   - [Simay Evin · 1/4, 1/6](#simay)
   - [Abdulkadir Öcal · 1/9](#abdulkadir)
   - [Sıla Taşan · 1/7, 1/11](#sila)

---

<a id="takim"></a>

## 1. Takım ve Görev Dağılımı

| Üye | Rol | Zimmetli Konular | Çıktı |
|---|---|---|---|
| Özge Sayınbaş | Dokümantasyon Yöneticisi & Çeviri Sorumlusu | 1/1 Giriş, 1/2 NLP ve LLM'ler, 1/10 Ünite Özeti | [`turkce_ceviri.md`](turkce_ceviri.md), [`cheat_sheet.md`](cheat_sheet.md), `README.md` (birleştirme), PR |
| Himmet Can Umutlu | Repo Kaptanı & Uygulama Kodlama Mühendisi | 1/3 Transformer'lar Neler Yapabilir?, 1/5 Görev Çözümleri, 1/8 LLM ile Çıkarım | [`pipeline_ve_inference.ipynb`](pipeline_ve_inference.ipynb) |
| Simay Evin | Sistem Mimarı: Teori & Algoritma | 1/4 Transformer'lar Nasıl Çalışır?, 1/6 Transformer Mimarileri | `README.md` → [Mimari ve Algoritma Analizi](#simay) |
| Abdulkadir Öcal | QA / Red-Teamer | 1/9 Önyargı ve Sınırlamalar | [`red_teaming_zafiyet.ipynb`](red_teaming_zafiyet.ipynb) |
| Sıla Taşan | Sınav Komiseri & Ölçme Değerlendirme | 1/7 Hızlı Quiz, 1/11 Sertifikasyon Sınavı | [`bolum_01_quiz_sinav.md`](bolum_01_quiz_sinav.md) |

<a id="klasor-yapisi"></a>

## 2. Klasör Yapısı

```text
NLP_2026/
└── bolum_01_transformer_modellerine_giris/
    ├── README.md                    # Takımın ana dokümanı 
    ├── turkce_ceviri.md             # Özge Sayınbaş: Bölüm 1'in bire bir Türkçe çevirisi
    ├── cheat_sheet.md               # Özge Sayınbaş: Transformer Cheat Sheet
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

<a id="notlar"></a>
