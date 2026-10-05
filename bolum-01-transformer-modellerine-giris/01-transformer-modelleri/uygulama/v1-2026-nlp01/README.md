# Uygulama: Pipeline ve Çıkarım (Inference)

**Hazırlayan:** Himmet Can Umutlu (Uygulama Kodlama Mühendisi)

## Çalıştırma

`main.ipynb` dosyasını Jupyter Notebook veya Google Colab ile açıp **Kernel → Restart & Run All** ile baştan sona çalıştırın. Gerekli kütüphaneler ilk satırda kurulur:

```bash
pip install transformers torch accelerate matplotlib
```

Modeller ilk çalıştırmada Hugging Face Hub'dan indirildiği için internet bağlantısı gerekir.

## Notebook İçeriği

| Bölüm | Konu | Ne yapılıyor? |
|---|---|---|
| Chapter 1/3: Pipelines | 1/3 | `pipeline()` ile duygu analizi, sıfır atışlı sınıflandırma ve maske doldurma |
| Chapter 1/5: Transformer Architectures | 1/5 | Encoder, decoder ve encoder-decoder mimarilerinin karşılaştırma tablosu |
| Chapter 1/8: Text Generation & Inference | 1/8 | GPT-2 ile greedy search, beam search, düşük/yüksek sıcaklık ve top-p stratejilerinin karşılaştırılması; sıcaklığın olasılık dağılımına etkisini gösteren grafik |

---

## 1/3 · Transformer'lar neler yapabilir?

### pipeline() Fonksiyonunun Çalışma Mantığı
- `pipeline()` 🤗 Transformers kütüphanesinin en temel nesnesidir.
- Bir modeli, ön işleme (preprocessing) ve son işleme (postprocessing) adımlarıyla birleştirir.
- Ham metni doğrudan girip anlaşılır bir yanıt almayı sağlar; iç detayları gizler.
- Pipeline'a metin geçtiğinde 3 ana adım gerçekleşir:
  1. Metin, modelin anlayacağı biçime **ön işlenir**.
  2. Ön işlenmiş girdiler **modele iletilir**.
  3. Modelin tahminleri, yorumlanabilir olması için **son işlenir**.
- Varsayılan olarak göreve uygun, önceden eğitilmiş bir model otomatik seçilir.
- Model, nesne ilk oluşturulduğunda indirilir ve **önbelleğe alınır**; sonraki çalıştırmalarda tekrar indirilmez.
- `pipeline()` metin, görüntü, ses ve çok modlu (multimodal) görevleri destekler.
- Hub'dan belirli bir model seçilebilir: `pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")`.

### Duygu Analizi (Sentiment Analysis)
- `pipeline("sentiment-analysis")` ile yapılır.
- Varsayılan model İngilizce duygu analizi için ince ayarlıdır.
- Çıktı: `{'label': 'POSITIVE', 'score': 0.96}` biçiminde etiket + güven skoru.
- Tek bir cümle veya **cümle listesi** (batch) verilebilir; liste hâlinde her cümle için ayrı sonuç döner.

### Zero-shot Sınıflandırma
- `pipeline("zero-shot-classification")` ile yapılır.
- Etiketsiz metni, **ince ayar gerektirmeden** sınıflandırır.
- `candidate_labels` parametresiyle etiket kümesini kullanıcı belirler (örn. `["education", "politics", "business"]`).
- Modele ait hazır etiketlere bağımlı kalmazsınız; istediğiniz her etiket için olasılık skoru döndürür.
- Etiket açıklamanın (annotation) zaman alıcı olduğu gerçek dünya senaryolarında güçlüdür.

### Metin Üretimi (Text Generation)
- `pipeline("text-generation")` ile yapılır.
- Bir istem (prompt) verilir; model kalan metni otomatik tamamlar (tahminli metin benzeri).
- Üretim **rastgelelik içerir**; aynı girdiyle aynı çıktı garanti değildir.
- `num_return_sequences`: kaç farklı dizi üretileceği.
- `max_length` / `min_length`: çıktının toplam uzunluğu.
- Hub'dan belirli bir model (örn. `HuggingFaceTB/SmolLM2-360M`) aynı pipeline'a yüklenebilir.
- Model Hub'daki widget ile model indirilmeden önce çevrimiçi test edilebilir.

### Adlandırılmış Varlık Tanıma (NER)
- `pipeline("ner", aggregation_strategy="simple")` ile yapılır.
- Girdi metninde kişi (PER), kuruluş (ORG), konum (LOC) gibi varlıkları bulur.
- Çıktı her varlık için `entity_group`, `score`, `word`, `start`, `end` bilgisi içerir.
- `aggregation_strategy="simple"` aynı varlığa ait kelimeleri birleştirir (örn. "Hugging" + "Face" → tek ORG).
- Ön işlemede kelimeler alt parçalara bölünebilir (örn. `Sylvain` → `S`, `##yl`, `##va`, `##in`); son işleme bunları yeniden gruplar.

### Diğer Pipeline'lar (Kısa Bakış)
- `fill-mask`: `<mask>` belirtecini doldurur; `top_k` kaç sonuç gösterileceğini belirler.
- `question-answering`: Bağlamdan bilgi çekerek soruyu yanıtlar (yanıtı kendisi üretmez).
- `summarization`: Metni, ana bilgileri koruyarak kısaltır.
- `translation`: Diller arası çeviri (örn. `Helsinki-NLP/opus-mt-fr-en`).
- Görüntü/ses: `image-classification`, `automatic-speech-recognition` gibi.

## 1/5 · Transformer'lar görevleri nasıl çözer?

### Dil Modeli Eğitiminin İki Ana Yaklaşımı
- **Maskelenmiş Dil Modelleme (MLM)** — Encoder (BERT):
  - Girdideki bazı belirteçler rastgele maskelenir (`[MASK]`).
  - Model, çevreleyen bağlamdan özgün belirteçleri tahmin eder.
  - **İki yönlü (bidirectional) bağlam** öğrenir: maskelenen kelimenin hem öncesine hem sonrasına bakar.
- **Nedensel Dil Modelleme (CLM)** — Decoder (GPT):
  - Dizideki **tüm önceki belirteçlere** dayanarak bir sonraki belirteci tahmin eder.
  - Yalnızca **soldan (önceki belirteçlerden)** gelen bağlamı kullanabilir.
  - Metin üretiminin temeli: her seferinde sıradaki kelimeyi tahmin eder.

### Encoder (BERT) vs Decoder (GPT) Ayrımı
| Özellik | Encoder (BERT) | Decoder (GPT) |
|---|---|---|
| Bağlam yönü | İki yönlü (sağ + sol) | Tek yönlü (sadece sol) |
| Eğitim hedefi | MLM + sonraki cümle tahmini | CLM (sonraki token) |
| Dikkat türü | Tam öz-dikkat | Maskelenmiş öz-dikkat (geleceğe bakamaz) |
| Tipik görevler | Sınıflandırma, NER, soru cevaplama | Metin üretimi, cümle tamamlama |
| Belirteçleme | WordPiece | BPE (Bayt Çifti Kodlaması) |

### BERT (Encoder) Teknik Detayları
- `[CLS]`: her dizinin başına eklenir; nihai çıktısı sınıflandırma başlığına gider.
- `[SEP]`: cümleleri ayırmak için kullanılır.
- Parça gömmesi (segment embedding): belirtecin cümle çiftinde 1. mi 2. mi cümlede olduğunu belirtir.
- İki ön eğitim amacı:
  1. **MLM**: belirli yüzde belirteç maskelenir, model tahmin eder (iki yönlülük "hile"sini çözer).
  2. **Sonraki cümle tahmini (NSP)**: B cümlesi A'yı takip ediyor mu? (`IsNext` / `NotNext`)
- Göreve göre üstüne bir **başlık (head)** eklenir: dizi sınıflandırma, belirteç sınıflandırma, span sınıflandırma.
- Gizli durumlar doğrusal dönüşümle **logits'e** çevrilir; çapraz entropi kaybı hesaplanır.

### GPT (Decoder) Teknik Detayları
- BPE ile belirteçleme; her belirtece konumsal kodlama (positional encoding) eklenir.
- Girdi gömmeleri birden çok **decoder bloğundan** geçer.
- Her blokta **maskelenmiş öz-dikkat**: gelecekteki belirteçlere bakılamaz (dikkat maskesiyle skorları 0 yapılır).
- Çıktı, gizli durumları logits'e çeviren **dil modelleme başlığına** gider.
- Etiket, bir sonraki belirteçtir; logits'ler bir adım **sağa kaydırılır** ve çapraz entropi kaybı hesaplanır.
- Ön eğitim amacı tamamen **nedensel dil modellemedir** (sonraki kelimeyi tahmin et).

### Üç Mimari Kategori (Özet)
- **Encoder-only** (BERT): derin anlama gerektiren görevler (sınıflandırma, NER, QA).
- **Decoder-only** (GPT, Llama): metin üretimi, kod üretimi.
- **Encoder-Decoder** (T5, BART): diziden diziye görevler (çeviri, özetleme).
- Seçim kuralı: iki yönlü bağlam → encoder; üretim → decoder; dizi→dizi → encoder-decoder.

### BART (Encoder-Decoder) Notu
- Kodlayıcısı BERT'e benzer; girdiyi **bozar** ve kod çözücüyle yeniden oluşturur.
- En iyi bozma stratejisi **metin doldurma (text infilling)**: bir span tek `[mask]` ile değiştirilir.
- Kodlayıcı çıktısı, kod çözücüye ek bağlam sağlar; kod çözücü otoregresif üretir.

## 1/8 · LLM'lerle çıkarım

### Temel Kavramlar
- **Çıkarım (inference)**: eğitilmiş LLM'in verilen istemden (prompt) insan benzeri metin üretme süreci.
- Model, yanıtı **her seferinde bir token** üretir; milyarlarca parametreden öğrendiği olasılıklara dayanır.
- **Dikkat (attention)**: bir sonraki token'ı tahmin ederken en ilgili kelimelere odaklanma yeteneği.
- **Bağlam uzunluğu**: modelin bir seferde işleyebildiği maksimum token sayısı ("dikkat süresi"/çalışma belleği).
  - Mimari, hesaplama kaynakları, girdi/çıktı karmaşıklığı ile sınırlıdır.

### İki Aşamalı Çıkarım Süreci

#### 1. Prefill (Ön Doldurma) Aşaması
- "Yanıt yazmadan önce paragrafın tamamını okumak" gibidir.
- Üç adım:
  1. **Tokenizasyon**: girdi metnini token'lara dönüştürme.
  2. **Gömme (embedding)**: token'ları anlam taşıyan sayısal vektörlere çevirme.
  3. **Başlangıç işleme**: gömme vektörlerini modelin sinir ağlarından geçirerek bağlamı anlama.
- **Hesaplama yoğun**: tüm girdi token'ları aynı anda (paralel) işlenir.
- İlk token gecikmesini (TTFT) büyük ölçüde bu aşama belirler.

#### 2. Decode (Kod Çözme / Üretim) Aşaması
- Gerçek metin üretiminin gerçekleştiği yer.
- **Otoregresif (autoregressive)**: her yeni token önceki tüm token'lara bağlıdır.
- Her yeni token için tekrarlanan adımlar:
  1. **Dikkat hesaplaması**: önceki token'lara geri bakma.
  2. **Olasılık hesaplaması**: her olası sonraki token'ın olabilirliğini bulma.
  3. **Token seçimi**: olasılıklara göre bir sonraki token'ı seçme.
  4. **Devam kontrolü**: üretmeye devam mı, durma mı (EOS)?
- **Bellek yoğun**: daha önce üretilen tüm token'lar ve ilişkileri takip edilir.

### Örnekleme Stratejileri (Token Seçimi)
- Model, sözlükteki her kelime için **ham logits** (işlenmemiş olasılıklar) üretir.

#### Temperature (Sıcaklık)
- "Yaratıcılık kadranı": olasılık dağılımını daraltır/genişletir.
- **> 1.0**: daha rastgele, yaratıcı, çeşitli seçimler.
- **< 1.0**: daha odaklı ve deterministik (keskin dağılım).
- **Greedy Search (açgözlü arama)**: her adımda en yüksek olasılıklı token'ı (argmax) seçme; sıcaklığın en deterministik uç noktası. Hızlı ama tekrara ve sığ sonuçlara yatkındır.

#### Top-k Filtreleme
- Yalnızca **en olası k** sonraki token dikkate alınır; kalanı elenir.
- Dağılım bu k token'a göre yeniden normalize edilir.

#### Top-p (Nucleus) Örnekleme
- Sabit sayı yerine, kümülatif olasılığı bir eşiğe (örn. %90) ulaşana kadar en olası kelimeler seçilir.
- Top-k'ya göre dağılımın şekline daha uyumludur; dinamik aday kümesi oluşturur.

### Tekrarı Yönetmek (Penaltılar)
- **Varlık cezası (presence penalty)**: daha önce görünmüş her token'a, sıklığına bakmaksızın **sabit** ceza.
- **Sıklık cezası (frequency penalty)**: bir token ne kadar çok kullanıldıysa ceza o kadar **artar**.
- Bu cezalar, diğer örnekleme stratejilerinden **önce** ham logits'lere uygulanır.

### Üretim Uzunluğunu Kontrol Etmek
- **Token sınırları**: min/maks token sayısı belirleme.
- **Durdurma dizileri**: üretim sonunu işaretleyen desenler (örn. `"\n\n"`).
- **EOS tespiti**: modelin yanıtı doğal yoldan bitirmesine izin verme (SmolLM2'de `<|im_end|>`).

### Beam Search (Hüzme Araması)
- Tek tek token kararı yerine **birden fazla aday yolu aynı anda** keşfeder (satranç gibi).
- Adımlar:
  1. Her adımda **birden fazla aday dizi** tut (tipik 5-10).
  2. Her aday için sonraki token olasılıklarını hesapla.
  3. En umut verici dizi + token kombinasyonlarını sakla.
  4. İstenen uzunluğa / durma koşuluna kadar devam et.
  5. **En yüksek toplam olasılıklı** diziyi seç.
- Daha tutarlı ve dilbilgisel olarak doğru metin üretir, ancak **daha fazla hesaplama** gerektirir.

### Pratik Zorluklar ve Optimizasyon
- **Performans metrikleri**:
  - TTFT (ilk token'a kadar geçen süre) — prefill'den etkilenir.
  - TPOT (çıktı token'ı başına süre) — üretim hızını belirler.
  - Throughput (verim) — aynı anda kaç istek.
  - VRAM kullanımı — genellikle birincil kısıt.
- **Bağlam uzunluğu maliyeti**:
  - Bellek kullanımı ∝ uzunluk² (kuadratik).
  - İşlem süresi ∝ uzunluk (doğrusal).
- **KV Cache (Anahtar-Değer Önbelleği)**: ara hesaplamaları depolayıp yeniden kullanır; tekrarlı hesabı azaltır, üretimi hızlandırır (bedeli ek bellek).
