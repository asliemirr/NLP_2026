# 🧠 Doğal Dil İçleme (NLP) — Açık Kaynak Sistem Mimarisi

SCÜ Şarkışla UBYO | Bilişim Sistemleri ve Teknolojileri Bölümü | 4. Sınıf

Bu depo, Doğal Dil İşleme dersinin **laboratuvar merkezi, yerelleştirilmiş dokümantasyon arşivi ve sızma testi (red-teaming) alanıdır.** Dersin ana omurgası **Hugging Face LLM Course** üzerinden yürütülmektedir.

> **📌 Kısa Özet:** Fork'la → Kendi kopyanda çalış → PR aç → Denetimden geç → Merge. Detaylar aşağıda.

---

## 📂 Depo Yapısı

Depo **bölüm bazlı** klasörlerden oluşur. Her bölüm altında **konu klasörleri**, her konu altında **4 içerik türü** bulunur:

```
nlp-dersi/
│
├── README.md
│
├── bolum-01-transformer-modellerine-giris/
│   ├── 01-attention-mekanizmasi/
│   │   ├── ceviri/
│   │   │   └── 2026-guz-kodbucuk.md
│   │   ├── sunum/
│   │   │   └── 2026-guz-kodbucuk.pdf
│   │   ├── uygulama/
│   │   │   └── v1-2026-kodbucuk/
│   │   │       ├── main.ipynb
│   │   │       └── README.md
│   │   ├── soru-cevap/
│   │   │   └── 2026-guz-kodbucuk.md
│   │   └── red-team/
│   │       └── 2026-guz-kodbucuk.md
│   │
│   ├── 02-transformer-mimarisi/
│   └── 03-tokenization/
│
├── bolum-02-huggingface-kutuphaneleri/
│   ├── 01-transformers-kutuphanesi/
│   ├── 02-datasets-kutuphanesi/
│   └── 03-tokenizers-kutuphanesi/
│
└── bolum-03-ince-ayar-finetuning/
    ├── 01-finetuning-temelleri/
    ├── 02-lora-peft/
    └── 03-degerlendirme/
```

**3 Katmanlı Yapı:**

| Katman | Ne? | Örnek |
|---|---|---|
| **1. Bölüm** | Müfredat bölümü (sabit) | `bolum-01-transformer-modellerine-giris/` |
| **2. Konu** | Bölüm içi alt konu | `01-attention-mekanizmasi/` |
| **3. İçerik Türü** | Çeviri / Sunum / Uygulama / Soru-Cevap / Red-Team | `ceviri/` |

---

## 📝 Dosya İsimlendirme Kuralları

### Çeviri, Sunum, Soru-Cevap, Red-Team için:

```
YIL-DONEM-TAKIMADI.uzanti
```

**Örnekler:**
- `2026-guz-kodbucuk.md`
- `2027-guz-yenitakim.pdf`

### Uygulama için:

```
vVERSIYON-YIL-DONEM-TAKIMADI/
```

**Örnekler:**
- `v1-2026-kodbucuk/`
- `v2-2027-yenitakim/`

> ⚠️ **Kabul edilen formatlar:** `.md`, `.qmd`, `.ipynb`, `.py`, `.pdf`
> ❌ **Kabul edilmeyen:** `.docx`, `.zip`, `.rar`, Google Docs linki

---

## 🔄 İş Akışı: Görevler Sisteme Nasıl Yüklenir?

Bu depoya **doğrudan dosya yükleme (Push) yetkiniz yoktur.** Görevlerinizi "Fork & Pull Request" akışıyla teslim edersiniz.

### Adım 1: Projeyi Kopyalayın (Fork)

1. Bu sayfanın sağ üst köşesindeki **Fork** butonuna tıklayın.
2. "Create Fork" diyerek deponun kopyasını kendi GitHub profilinizde oluşturun.

### Adım 2: Kendi Kopyanızda Çalışın

1. Kendi profilinizdeki kopyaya gidin.
2. Bilgisayarınıza indirin (`git clone`) veya GitHub arayüzünden dosya ekleyin.
3. **Doğru klasöre gidin** ve dosyanızı ekleyin.

**📁 Nereye Koyacağım? (Karar Ağacı)**

```
Hangi bölümdeyim?
  → bolum-01-transformer-modellerine-giris/ (veya 02, 03)

Hangi konuyu işliyorum?
  → 01-attention-mekanizmasi/ (veya diğer konular)

Hangi tür dosya hazırlıyorum?
  → Çeviri ise: ceviri/
  → Sunum ise: sunum/
  → Uygulama ise: uygulama/
  → Soru-cevap ise: soru-cevap/
  → Red-team ise: red-team/

Dosya ismim ne olacak?
  → YIL-DONEM-TAKIMADI.md (örn: 2026-guz-kodbucuk.md)
```

### Adım 3: Lokal Test (Zorunlu)

Kodunuzu `.ipynb` veya `.py` olarak teslim etmeden önce:

```bash
# Jupyter notebook'u çalıştırın
jupyter notebook main.ipynb

# Veya Python scriptini test edin
python main.py
```

**Çalışmayan kod PR aşamasında reddedilir.**

### Adım 4: Katkı Talebi (Pull Request) Gönderin

1. Kendi fork'unuzda işinizi bitirin (**Commit & Push**).
2. Kendi deponuzun ana sayfasına gelin.
3. **"Contribute"** → **"Open Pull Request"** seçin.
4. **Hedef:** `mesutpolatgil/nlp-dersi` → `main` dalı
5. **PR Başlığı:** `Hafta XX - [Takım Adı] Teslimi`
   - Örnek: `Hafta 03 - KodBucuk Teslimi`
6. **Açıklama:** Kısaca ne yaptığınızı yazın (3-5 satır).

### Adım 5: Denetim (Code Review)

1. PR açtığınızda **hoca ve asistan** bildirim alır.
2. **48 saat içinde** PR'ınız incelenir.
3. Eksik veya hatalı yerler varsa **PR üzerinden yorum** yapılır.
4. Gerekli düzeltmeleri yaptıktan sonra tekrar push edin.
5. Onaylanan PR merge edilir.

> ⚠️ **3 tur düzeltmeden sonra hâlâ eksikse** PR reddedilir ve takım o hafta **0 alır.**

---

## 👥 Takım Rolleri ve Depo Katkısı

Sınıftaki **4 kişilik Yapay Zeka Ar-Ge Takımları**, bu repoyu aşağıdaki görev dağılımına göre besleyecektir:

| # | Rol | Sorumluluk |
|---|---|---|
| **1** | **Dokümantasyon Yöneticisi** | Hugging Face orijinal metnini çevirir, yerelleştirilmiş teknik rehber ekler, PR'ı açar |
| **2** | **Sistem Mimarı** | Algoritmanın temelini ve icat edilme nedenini tahtada sezgisel olarak anlatır |
| **3** | **QA & Adli Bilişim Uzmanı** | Kodu kasten bozar, sistemi zorlar (hallucination, prompt injection) — canlı test eder |
| **4** | **Sınav Komiseri** | Ünite sonu quizlerini akademik formata getirir, sınıfı test eder |

> ⚠️ **Altın Kural:** Her öğrenci sunum esnasında sahneye çıkmak ve **kendi rolüyle ilgili** konuşmak zorundadır. Arkada sessiz kalan üye değerlendirmeye alınmaz.

---

## 💬 Commit Mesajı Formatı

```
hafta-XX: [TakımAdı] kısa açıklama
```

**Örnekler:**
```
hafta-03: KodBucuk attention cevirisi eklendi
hafta-03: KodBucuk tokenization notebook eklendi
hafta-03: KodBucuk red-team raporu eklendi
```

> ❌ **Kabul edilmeyen:** `asdf`, `update`, `değişiklik`, `final`, `son hali`

---

## ✅ PR Öncesi Kontrol Listesi

PR açmadan önce şunları kontrol edin:

- [ ] Doğru bölüm ve konu klasöründe miyim?
- [ ] Doğru içerik türünde miyim? (`ceviri/`, `sunum/`, `uygulama/`, `soru-cevap/`, `red-team/`)
- [ ] Dosya ismi doğru formatta mı? (`2026-guz-takimadi.md`)
- [ ] Uygulama klasöründe `README.md` var mı?
- [ ] Kodum **lokal olarak çalışıyor mu?** (Jupyter/Python test edildi mi?)
- [ ] Çeviri dosyasının başında **yasal uyarı** var mı?
- [ ] `.docx`, `.zip` gibi yasaklı format kullanmadım, değil mi?
- [ ] Commit mesajı standart formatta mı?
- [ ] PR başlığı formatı doğru mu?
- [ ] Başka takımın dosyasına **dokunmadım**, değil mi?

---

## 📎 Örnek Teslim

**Senaryo:** 1. hafta, "KodBucuk" takımı, "Attention Mekanizması" konusu.

**Ekleyeceği dosyalar:**

```
bolum-01-transformer-modellerine-giris/01-attention-mekanizmasi/
  ├── ceviri/
  │   └── 2026-guz-kodbucuk.md
  ├── sunum/
  │   └── 2026-guz-kodbucuk.pdf
  ├── uygulama/
  │   └── v1-2026-kodbucuk/
  │       ├── main.ipynb
  │       └── README.md
  ├── soru-cevap/
  │   └── 2026-guz-kodbucuk.md
  └── red-team/
      └── 2026-guz-kodbucuk.md
```

**Commit mesajları:**
```
hafta-01: KodBucuk attention cevirisi eklendi
hafta-01: KodBucuk slayt eklendi
hafta-01: KodBucuk notebook eklendi
hafta-01: KodBucuk red-team raporu eklendi
```

**PR başlığı:**
```
Hafta 01 - KodBucuk Teslimi
```

---

## 📅 Teslim Kuralları

| Kural | Detay |
|---|---|
| **Deadline** | Sunumdan önceki **Cuma 23:59** |
| **Geç Teslim** | 24 saat içinde %20 kırılır, sonrası kabul edilmez |
| **PR Hedefi** | `main` dalı |
| **PR Başlığı** | `Hafta XX - [Takım Adı] Teslimi` |
| **Commit Formatı** | `hafta-XX: TakimAdi ne-yaptim` |
| **Zorunlu Test** | Kod lokal olarak çalışmalı |
| **Kabul Edilen Formatlar** | `.md`, `.qmd`, `.ipynb`, `.py`, `.pdf` |
| **Yasaklı Formatlar** | `.docx`, `.zip`, `.rar`, Google Docs linki |

---

## 📜 Yasal Uyarı ve Telif

Bu depo altındaki içerikler **eğitim amacıyla** hazırlanmaktadır.

Öğrenci takımları tarafından hazırlanan **tüm çeviri dosyalarının en üstünde** şu ibare yer almak zorundadır:

> **Uyarı:** Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

**Yasal uyarı olmayan çeviri PR'ları reddedilir.**

---

## 🎯 İlk Göreviniz

1. **Takım kurun** (4 kişi, sınıf temsilcisi koordine eder).
2. **Konu seçin** (14 modülden biri).
3. **GitHub Issues** sekmesinde, sınıf temsilcisinin açtığı *"Dönem İçi Konu Dağılım Listesi"* başlığı altına **yorum olarak** yazın.
4. İlk sunumunuzdan önceki **Cuma 23:59**'a kadar ilk PR'ınızı açın.

---

## 📞 İletişim

- **Teknik sorular:** GitHub Discussions
- **Acil durumlar:** [Hoca/Asistan e-posta]
- **PR incelemesi:** Hoca + asistan

---

Tüm takımlara mühendislik simülasyonunda başarılar dilerim.
**Modeliniz overfit olmasın, loss'unuz düşsün!**
