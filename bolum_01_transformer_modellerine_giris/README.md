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

<a id="notlar"></a>

## 4. Teorik Notlar ve Kod Açıklamaları

Bu bölümde her takım üyesinin kendi konularına ait teorik notları ve kod açıklamaları yer alır. Bölümün Türkçe çevirisi [`turkce_ceviri.md`](turkce_ceviri.md), hızlı başvuru kartı [`cheat_sheet.md`](cheat_sheet.md) dosyasındadır.

<a id="ozge"></a>

### 4.1 Özge Sayınbaş · Dokümantasyon Yöneticisi & Çeviri Sorumlusu

**Konular:** 1/1 Giriş, 1/2 Doğal Dil İşleme ve Büyük Dil Modelleri, 1/10 Ünite Özeti

<a id="k1-1"></a>

#### 1/1 · Giriş

> Kaynak: [huggingface.co/learn/llm-course/chapter1/1](https://huggingface.co/learn/llm-course/chapter1/1)

Bu kurs, Hugging Face ekosistemindeki Transformers, Datasets, Tokenizers ve Accelerate kütüphaneleri ile Hugging Face Hub'ı kullanarak büyük dil modellerini (LLM) ve doğal dil işlemeyi (NLP) öğretir. İlk bölümde, metin üretimi ve sınıflandırma gibi NLP görevlerinin `pipeline()` fonksiyonuyla nasıl çözüldüğü, Transformer mimarisi ve encoder, decoder ve encoder-decoder mimarilerinin farkları ile kullanım alanları ele alınır.

**Doğal Dil İşleme (NLP)**, bilgisayarların insan dilini anlamasını, yorumlamasını ve üretmesini sağlayan geniş bir alandır. Duygu analizi (sentiment analysis), adlandırılmış varlık tanıma (Named Entity Recognition, NER) ve makine çevirisi (machine translation) bu alanın görevlerine örnektir.

> **Örnek:** Bir ürün yorumunun olumlu mu olumsuz mu olduğunu bulmak (duygu analizi) ya da bir cümleyi Türkçeden İngilizceye çevirmek (makine çevirisi) birer NLP görevidir.

**Büyük Dil Modelleri (Large Language Models, LLM)**, NLP modellerinin güçlü bir alt kümesidir. Çok büyük boyutları ve çok büyük miktarda veriyle eğitilmeleri sayesinde, göreve özgü çok az eğitimle birçok farklı dil görevini yerine getirebilirler. GPT, Llama ve Claude bu modellere örnektir.

> **Örnek:** Eskiden çeviri için bir model, özet çıkarmak için başka bir model gerekirdi. Bir LLM ise tek başına hem çeviri yapabilir hem özet çıkarabilir hem de soruları yanıtlayabilir.

Kısacası NLP geniş bir alandır; LLM ise bu alanın en güçlü araçlarından biridir. LLM'lerle verimli çalışabilmek için NLP'nin temel kavramlarını bilmek gerekir.

<a id="k1-2"></a>

#### 1/2 · Doğal Dil İşleme ve Büyük Dil Modelleri

> Kaynak: [huggingface.co/learn/llm-course/chapter1/2](https://huggingface.co/learn/llm-course/chapter1/2)

**NLP**, dilbilim ile makine öğrenmesinin kesişiminde yer alır ve kelimeleri bağlamlarıyla birlikte anlamayı amaçlar. Başlıca görevleri şunlardır:

- **Cümleyi bütün olarak sınıflandırmak:** "Bu film harikaydı!" → olumlu yorum · "Bedava telefon kazandınız!" → spam
- **Cümledeki kelimeleri sınıflandırmak:** "Ayşe Ankara'da çalışıyor." → Ayşe = kişi, Ankara = yer
- **Metin üretmek:** "Bugün hava çok …" → model istemi (prompt) tamamlar
- **Metinden yanıt çıkarmak:** Metin: "Ali 2010'da doğdu." · Soru: "Ali ne zaman doğdu?" → 2010
- **Yeni metin üretmek:** Çeviri ("Good morning" → "Günaydın") veya uzun bir metni kısa bir özete dönüştürmek

NLP yalnızca yazıyla sınırlı değildir; bir ses kaydını yazıya dökmek (konuşma tanıma, speech recognition) ve bir fotoğrafı betimlemek de bu alana girer.

**LLM'ler**, devasa metin verisiyle eğitilmiş ve göreve özgü eğitim olmadan birçok dil görevini yapabilen modellerdir. Milyarlarca parametreye sahiptirler, istemde verilen örneklerden öğrenebilirler (bağlam içi öğrenme, in-context learning) ve büyüdükçe açıkça programlanmamış ya da öngörülmemiş yetenekler gösterebilirler (ortaya çıkan yetenekler, emergent abilities).

> **Örnek (bağlam içi öğrenme):** Modele "mutlu → olumlu, üzgün → olumsuz, sevinçli → ?" yazılırsa, model kalıbı anlayıp "olumlu" cevabını verir. Bunun için modeli yeniden eğitmeye gerek yoktur.

**Sınırlamaları:**

- **Halüsinasyon (hallucination):** Yanlış bilgiyi emin bir şekilde üretebilir. *Örnek: Var olmayan bir kitabı gerçekmiş gibi anlatmak.*
- **Gerçek anlama eksikliği:** Dünyayı gerçek anlamda kavrayamaz; tamamen istatistiksel örüntülerle çalışır.
- **Önyargı (bias):** Eğitim verisindeki önyargıları tekrarlayabilir. *Örnek: Meslekleri belirli bir cinsiyetle eşleştirmek.*
- **Sınırlı bağlam penceresi (context window):** Çok uzun bir metnin tamamını aynı anda dikkate alamaz.
- **Yüksek hesaplama ihtiyacı:** Eğitmek ve çalıştırmak için güçlü donanım gerekir.

**Dil işlemenin zorluğu:** Bilgisayarlar metni insanlar gibi anlayamaz; metnin önce modelin işleyebileceği bir biçime dönüştürülmesi gerekir.

> **Örnek:** "Acıktım" cümlesini okuyan biri anlamını hemen kavrar; "Acıktım" ile "Üzgünüm" cümlelerinin ne kadar benzediğini de kolayca söyler. Bir model için bu zordur.

Çok anlamlılık, kültürel bağlam, iğneleme (sarkazm) ve mizah, LLM'ler için bile hâlâ zor konulardır. *Örneğin yağmurda ıslanan birinin "Ne güzel hava!" demesindeki iğnelemeyi anlamak.*

<a id="k1-10"></a>

#### 1/10 · Ünite Özeti

> Kaynak: [huggingface.co/learn/llm-course/chapter1/10](https://huggingface.co/learn/llm-course/chapter1/10)

**Doğal Dil İşleme ve LLM'ler:** NLP, metni sınıflandırmaktan yeni metin üretmeye kadar çok geniş bir görev yelpazesini kapsar. LLM'ler, devasa miktarda metinle eğitilmiş güçlü modellerdir ve tek bir mimari içinde birden fazla görevi yerine getirebilir. Bu yeteneklerine rağmen halüsinasyon ve önyargı gibi sınırlamaları vardır.

> **Örnek:** Aynı LLM'ye önce bir yorumun olumlu mu olumsuz mu olduğunu sorup ardından bir metni özetletebilirsiniz; iki iş için ayrı model gerekmez. Ancak model bazen yanlış bir bilgiyi doğruymuş gibi söyleyebilir (halüsinasyon) ya da eğitim verisindeki önyargıları tekrarlayabilir.

**Transformer modelleri**, günümüz NLP'sinin ve LLM'lerin temelini oluşturur. Hugging Face'in `pipeline()` fonksiyonu, ön eğitimli bu modelleri metin sınıflandırma, soru yanıtlama, metin üretimi, özetleme, çeviri, konuşma tanıma ve görüntü sınıflandırma gibi görevlerde kolayca kullanmayı sağlar.

> **Örnek:**
> ```python
> from transformers import pipeline
> classifier = pipeline("sentiment-analysis")
> classifier("I've been waiting for a HuggingFace course my whole life.")
> # [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
> ```
> Birkaç satır kodla hazır bir model yüklenir ve cümlenin olumlu olduğu bulunur.

Transformer'ların temelinde **dikkat (attention) mekanizması** vardır: Model, bir kelimeyi işlerken cümledeki ilgili kelimelere odaklanır.

> **Örnek:** "Fransa'nın başkenti …" cümlesinde model, sonraki kelimenin "Paris" olduğunu tahmin etmek için en çok "Fransa" ve "başkent" kelimelerine dikkat eder.

**Transfer öğrenme (transfer learning)** sayesinde, büyük veriyle önceden eğitilmiş bir model daha az veriyle yeni bir göreve uyarlanabilir.

> **Örnek:** İngilizce metinlerle önceden eğitilmiş bir model, bilimsel makalelerle kısa bir ek eğitimden geçirilerek bilim alanında uzmanlaşmış bir modele dönüştürülebilir. Sıfırdan eğitmeye göre çok daha az zaman, veri ve maliyet gerekir.

Transformer modelleri üç ana türe ayrılır:

| Tür | Ne yapar? | Örnek model | Örnek görev |
|---|---|---|---|
| **Encoder** | Metni anlar | BERT | Bir yorumun olumlu mu olumsuz mu olduğunu bulmak |
| **Decoder** | Metin üretir | GPT, LLaMA | Bir hikâyenin devamını yazmak, sohbet etmek |
| **Encoder-decoder** | Metni başka bir metne dönüştürür | BART, T5 | Bir metni çevirmek veya özetlemek |

**Modern LLM gelişmeleri:** Modeller boyut ve yetenek bakımından giderek büyümektedir; bu büyümeye ölçekleme yasaları (scaling laws) yön verir. Modern LLM'ler iki aşamada eğitilir: önce çok büyük veriyle ön eğitim (pretraining), ardından talimatlara uymak üzere talimat ayarı (instruction tuning). Daha uzun metinleri işleyebilmek için özelleşmiş dikkat mekanizmaları geliştirilmiştir.

**Pratik uygulamalar:** Ön eğitimli modeller Hugging Face Hub'dan bulunup kullanılabilir, Inference API ile doğrudan tarayıcıda denenebilir. Önemli olan, göreve en uygun modeli seçmektir.

> **Örnek:** Metin üretecek bir model arıyorsanız Hub'da "text-generation" etiketine tıklayıp listelenen modellerden birini seçebilir, indirmeden önce sayfasındaki bileşenle (widget) deneyebilirsiniz.

---

<a id="himmet"></a>

### 4.2 Himmet Can Umutlu · Uygulama Kodlama Mühendisi

**Konular:** 1/3 Transformer'lar neler yapabilir?, 1/5 Transformer'lar görevleri nasıl çözer?, 1/8 LLM'lerle çıkarım · **Notebook:** [`pipeline_ve_inference.ipynb`](pipeline_ve_inference.ipynb)

<!-- HIMMET CAN UMUTLU: Notebook'taki her bölüm için 1-2 cümlelik açıklamayı tablolara ekle. -->

<a id="k1-3"></a>

#### 1/3 · Transformer'lar neler yapabilir?

| Notebook bölümü | Açıklama |
|---|---|
| Sentiment Analysis | <!-- açıklama --> |
| Zero-shot Classification | <!-- açıklama --> |
| Text Generation | <!-- açıklama --> |
| NER | <!-- açıklama --> |

<a id="k1-5"></a>

#### 1/5 · Transformer'lar görevleri nasıl çözer?

<!-- HIMMET CAN UMUTLU: 1/5 ile ilgili notebook bölümlerini ve açıklamalarını buraya ekle. -->

<a id="k1-8"></a>

#### 1/8 · LLM'lerle çıkarım

| Notebook bölümü | Açıklama |
|---|---|
| Temperature, Top-k, Top-p | <!-- açıklama + grafik yorumu --> |
| Greedy Search vs. Beam Search | <!-- açıklama --> |

---

<a id="simay"></a>
<a id="mimari-analiz"></a>

### 4.3 Simay Evin · Sistem Mimarı (Mimari ve Algoritma Analizi)

#### 1/4 · Transformer'lar nasıl çalışır?

##### 1. Transformer Mimarisi ve Blok Şeması

Transformer, özellikle doğal dil işleme (NLP) problemlerinde kullanılan ve diziler arasındaki ilişkileri **attention (dikkat)** mekanizması ile öğrenen bir sinir ağı mimarisidir. Geleneksel RNN ve LSTM yapılarından farklı olarak, bir dizideki elemanları yalnızca sırayla işlemek yerine tokenlar arasındaki ilişkileri aynı yapı içinde değerlendirebilir.

Bir Transformer yapısı temel olarak iki ana bloktan oluşur:

- **Encoder:** Girdiyi işler ve bağlamsal bir temsil oluşturur.
- **Decoder:** Encoder'dan gelen temsili ve daha önce üretilen çıktıları kullanarak yeni çıktı tokenlarını üretir.

Transformer modellerinde bu iki yapı birlikte kullanılabildiği gibi yalnızca encoder veya yalnızca decoder da kullanılabilir.

Aşağıdaki şema Transformer mimarisinin temel akışını göstermektedir:

```text
                    GİRDİ
          "Ben bugün okula gidiyorum"
                      │
                      ▼
          Token + Positional Encoding
                      │
                      ▼
        ┌──────────────────────────┐
        │         ENCODER          │
        │                          │
        │  Multi-Head             │
        │  Self-Attention         │
        │          │               │
        │          ▼               │
        │      Add & Norm          │
        │          │               │
        │          ▼               │
        │     Feed Forward         │
        │          │               │
        │          ▼               │
        │      Add & Norm          │
        └────────────┬─────────────┘
                     │
                     ▼
          Bağlamsal Temsil / Context
                     │
                     ▼
        ┌──────────────────────────┐
        │         DECODER          │
        │                          │
        │  Masked Multi-Head      │
        │  Self-Attention         │
        │          │               │
        │          ▼               │
        │      Add & Norm          │
        │          │               │
        │          ▼               │
        │  Encoder-Decoder         │
        │  Attention               │
        │          │               │
        │          ▼               │
        │      Add & Norm          │
        │          │               │
        │          ▼               │
        │     Feed Forward         │
        │          │               │
        │          ▼               │
        │      Add & Norm          │
        └────────────┬─────────────┘
                     │
                     ▼
                   ÇIKTI
```

Şemanın sezgisel akışı şu şekildedir:

1. Girdi metni önce **tokenlara** ayrılır.
2. Tokenlara sıralarını göstermek için **positional encoding (konum bilgisi)** eklenir.
3. **Encoder**, tokenlar arasındaki ilişkileri Self-Attention ile değerlendirerek bağlamsal bir temsil oluşturur.
4. **Decoder**, encoder'dan gelen bu temsil ile daha önce üretilen tokenları kullanarak çıktı üretir.
5. Decoder gelecekteki tokenları görmemek için **Masked Self-Attention** kullanır.

> Encoder ve decoder blokları gerçek Transformer yapısında birden fazla kez tekrar edilebilir.

---

##### 2. Self-Attention Mekanizması

Transformer mimarisinin temel bileşenlerinden biri **Self-Attention** mekanizmasıdır. Self-Attention, bir token işlenirken aynı dizideki diğer tokenların o token için ne kadar önemli olduğunu hesaplar.

Örneğin:

> "Kedi sütü içti."

Model **"içti"** tokenını işlerken "kedi" ve "sütü" tokenlarının bu kelimeyle ne kadar ilişkili olduğunu dikkate alabilir. Böylece kelime yalnızca kendi başına değil, cümledeki bağlamıyla birlikte temsil edilir.

Self-Attention işleminde her token için üç farklı temsil oluşturulur:

- **Q — Query (Sorgu):** Tokenın hangi bilgiyi aradığını temsil eder.
- **K — Key (Anahtar):** Tokenın hangi bilgiyle eşleşebileceğini temsil eder.
- **V — Value (Değer):** Tokenın taşıdığı asıl bilgiyi temsil eder.

Sezgisel olarak:

```text
Q → "Ne arıyorum?"
K → "Bende hangi ipucu var?"
V → "Taşıdığım bilgi nedir?"
```

Self-Attention işlemi şu formülle ifade edilir:

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Formülün adım adım anlamı:

1. **Q ile K karşılaştırılır.**  
   \(QK^T\) işlemi ile tokenlar arasındaki ilişki skorları hesaplanır.

2. **Skorlar ölçeklenir.**  
   Elde edilen değerler \(\sqrt{d_k}\)'ya bölünerek skorların aşırı büyümesi önlenir.

3. **Softmax uygulanır.**  
   Skorlar dikkat ağırlıklarına dönüştürülür. Böylece hangi tokena ne kadar dikkat edileceği belirlenir.

4. **V değerleri ağırlıklandırılır.**  
   Hesaplanan dikkat ağırlıkları Value değerleri ile birleştirilir ve token için yeni, bağlama duyarlı bir temsil oluşturulur.

Kısaca:

```text
Q × Kᵀ
   ↓
İlişki skorları
   ↓
÷ √dk
   ↓
Ölçekleme
   ↓
Softmax
   ↓
Dikkat ağırlıkları
   ↓
× V
   ↓
Bağlamsal temsil
```

---

##### 3. Transformer ile RNN/LSTM Karşılaştırması

RNN ve LSTM tabanlı yapılarda dizideki elemanlar genellikle **ardışık (sequential)** olarak işlenir. Bir zaman adımındaki hesaplama, önceki zaman adımından gelen bilgiye bağlıdır.

Basitleştirilmiş RNN/LSTM akışı:

```text
Token 1 → Token 2 → Token 3 → Token 4
```

Bu yapı nedeniyle bir adım tamamlanmadan sonraki adımın hesaplanması zorlaşır ve eğitim sırasında paralelleştirme sınırlı kalır.

Transformer mimarisinde ise Self-Attention işlemleri büyük ölçüde **matris işlemleri** üzerinden gerçekleştirildiği için bir dizideki tokenlar arasındaki ilişkiler aynı anda hesaplanabilir.

Basitleştirilmiş Transformer yaklaşımı:

```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼→ Self-Attention
Token 4 ─┘
```

Bu nedenle Transformer:

- GPU gibi paralel işlem gücü yüksek donanımlardan daha iyi yararlanabilir.
- Eğitim sırasında daha fazla işlemi paralel gerçekleştirebilir.
- Büyük veri kümeleri ve büyük modeller üzerinde daha verimli ölçeklenebilir.

> **Önemli not:** Decoder-only modeller, çıktı üretimi sırasında tokenları yine sırayla üretir. Transformer'ın RNN/LSTM'ye göre paralelleştirme avantajı özellikle eğitim aşamasında belirgindir.
>
> #### 1/6 · Transformer Mimarileri

##### 4. Transformer Mimari Aileleri

Transformer tabanlı modeller üç temel mimari aile altında incelenebilir:

| Mimari | Yapı | Bağlam Kullanımı | Temel Kullanım | Örnek |
|---|---|---|---|---|
| **Encoder-only** | Yalnızca Encoder | Girdinin iki tarafındaki bağlamı kullanabilir | Metni anlama, sınıflandırma | BERT |
| **Decoder-only** | Yalnızca Decoder | Önceki tokenlara bakarak sonraki tokenı üretir | Metin üretimi | GPT |
| **Encoder-Decoder** | Encoder + Decoder | Girdiyi işler ve yeni bir çıktı dizisi üretir | Çeviri, özetleme | T5 |

###### 4.1. Encoder-only — BERT

Encoder-only modeller girdinin tamamını bağlam içinde değerlendirmeye odaklanır. Bu nedenle metni **anlamaya** yönelik görevlerde kullanışlıdır.

Örnek görevler:

- Metin sınıflandırma
- Duygu analizi
- Varlık tanıma (Named Entity Recognition)

Basit şekilde:

```text
Metin
  ↓
Encoder
  ↓
Bağlamsal temsil
  ↓
Sınıflandırma / Anlama
```

**BERT**, encoder-only mimarinin bilinen örneklerinden biridir.

---

###### 4.2. Decoder-only — GPT

Decoder-only modeller temel olarak **bir sonraki tokenı tahmin ederek metin üretmeye** odaklanır.

Örneğin modelin girdisi:

> "Bugün hava çok"

ise model bir sonraki token olarak:

> "güzel"

gibi bir kelime üretebilir.

Decoder, bir tokenı üretirken henüz oluşmamış gelecekteki tokenlara bakmaz; yalnızca daha önceki tokenları kullanır.

Basit şekilde:

```text
"Bugün"
   ↓
"Bugün hava"
   ↓
"Bugün hava çok"
   ↓
"Bugün hava çok güzel"
```

**GPT**, decoder-only mimarinin bilinen örneklerinden biridir.

---

###### 4.3. Encoder-Decoder — T5

Encoder-Decoder modeller iki aşamalı çalışır:

1. **Encoder** girdiyi işler ve anlamlı bir temsil oluşturur.
2. **Decoder** bu temsilden yararlanarak yeni bir çıktı dizisi üretir.

Örnek:

```text
"I love cats."
      ↓
   Encoder
      ↓
Bağlamsal temsil
      ↓
   Decoder
      ↓
"Kedileri seviyorum."
```

Bu yapı özellikle bir dizinin başka bir diziye dönüştürüldüğü görevlerde kullanılır.

Örnek görevler:

- Makine çevirisi
- Metin özetleme
- Metinden metne dönüşüm

**T5**, encoder-decoder mimarisinin bilinen örneklerinden biridir.

---

##### 5. Genel Özet

Transformer mimarisinin temel gücü, tokenlar arasındaki ilişkileri **Self-Attention** mekanizması ile doğrudan hesaplayabilmesidir. Self-Attention içinde kullanılan **Query, Key ve Value** yapıları sayesinde model hangi tokenların birbirleriyle daha ilişkili olduğunu belirler ve bağlama duyarlı temsiller üretir.

RNN ve LSTM yapılarının ardışık işlem bağımlılığına karşılık Transformer, attention hesaplamalarını matris işlemleriyle gerçekleştirdiği için eğitim sırasında paralelleştirmeye daha uygundur.

Transformer tabanlı modellerin temel mimari aileleri ise:

- **BERT → Encoder-only → Anlama**
- **GPT → Decoder-only → Üretme**
- **T5 → Encoder-Decoder → Girdiyi işleyip yeni çıktı üretme**

şeklinde özetlenebilir.

<a id="sila"></a>

### 4.5 Sıla Taşan · Sınav Komiseri & Ölçme Değerlendirme

**Konular:** 1/7 Kısa sınav, 1/11 Sertifika sınavı · **Dosya:** [`bolum_01_quiz_sinav.md`](bolum_01_quiz_sinav.md)

<!-- SILA TAŞAN: Sınav çalışmanla ilgili kısa notunu buraya ekle. -->

1/7 ve 1/11 sorularının Türkçe hâli, özgün akademik sorular ve çözüm anahtarı [`bolum_01_quiz_sinav.md`](bolum_01_quiz_sinav.md) dosyasındadır.
