# Bölüm 0: NLP Laboratuvar Altyapısı, Donanım Mimarisi ve Çalışma Zamanı (Runtime) Zafiyet Analizi

**Ders:** Doğal Dil İşleme (NLP) · 4. Sınıf Mühendislik Çekirdeği  
**Öğretim Üyesi:** Dr. Mesut Polatgil  
**Sorumlu Araştırmacı:** Mehmet Can Efe  
**Resmî Müfredat Referansı:** [Hugging Face LLM Course - Chapter 0 (Setup / Introduction)](https://huggingface.co/learn/llm-course/chapter0/1)  

---

## 1. Giriş ve Mühendislik Felsefesi

Bu çalışma, hazır bir API veya `import transformers` çağrısından ibaret yüzeysel bir kurulum kılavuzu **değildir**. 4. sınıf bilgisayar/yazılım mühendisliği seviyesinde bir araştırmacının; Büyük Dil Modellerini (LLM) çalıştıracağı donanım katmanını, bellek hiyerarşisini, tensör veri tiplerini ve çalışma zamanı (runtime) sınırlarını derinlemesine kavramasını amaçlar.

Hugging Face resmi müfredatı Windows işletim sistemi adımlarını atlayıp yalnızca genel Colab/Linux yönergeleri vermektedir. Bu doküman, hem eksik bırakılan platform dinamiklerini (Windows NTFS symlink yapısı, execution policy, UTF-8 konsol bayt akışı) tamamlamakta hem de modellerin hangi matematiksel ve donanımsal sınırlarda çöktüğünü **Adli Bilişim (Forensics & Red-Teaming)** metodolojisiyle ortaya koymaktadır.

---

## 2. Donanım ve Bellek Mimarisi Matematiği

Bir dil modelini yerel donanıma veya sunucuya indirmeden önce, sistem mimarının yapması gereken ilk iş **bellek ayak izini (memory footprint)** analitik olarak hesaplamaktır.

### 2.1 Model Ağırlıklarının VRAM/RAM İhtiyacı Formülü

Modelin parametre sayısı $P$, parametre başına kullanılan bit genişliği $b$ olmak üzere, yalnızca model ağırlıklarının disk ve VRAM'deki statik boyutu:

$$M_{\text{weights}} = P \times \left(\frac{b}{8}\right) \text{ Bayt}$$

Çalışma zamanında (Inference Runtime) aktivasyon tensörleri, KV-Önbellek (KV-Cache) ve CUDA bağlamı için en az **%20 ek tampon payı** gerekir:

$$M_{\text{total\_inference}} \approx M_{\text{weights}} \times 1.20$$

#### Parametre Büyüklüklerine Göre Bellek Tüketim Matrisi:

| Model Parametresi ($P$) | FP32 (32-bit Float, 4B) | FP16 / BF16 (16-bit, 2B) | INT8 (Quantized, 1B) | INT4 (AWQ/GPTQ, 0.5B) | Minimum Donanım Gereksinimi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0.1B (BERT-Base / 110M)** | ~440 MB | ~220 MB | ~110 MB | ~55 MB | Standart CPU / 4 GB RAM |
| **0.5B (DistilGPT2 / Qwen-0.5B)** | ~2.0 GB | ~1.0 GB | ~500 MB | ~250 MB | Standart CPU / 8 GB RAM |
| **7B (Llama-3-8B / Mistral-7B)** | **~28.0 GB** | **~14.0 GB** | **~7.0 GB** | **~3.5 GB** | 16 GB VRAM (FP16) veya 8 GB VRAM (INT4) |
| **70B (Llama-3-70B)** | ~280 GB | ~140 GB | ~70 GB | ~35 GB | $2 \times \text{A100 (80GB)}$ veya $1 \times \text{H100}$ |

> ** Mühendislik Çıkarımı:** 8 GB tüketici sınıfı (RTX 3070 / RTX 4060) bir ekran kartında 7B büyüklüğünde bir model FP16 (14 GB) olarak yüklenmeye çalışıldığında **anında CUDA Out-Of-Memory (OOM)** hatasıyla çöker. Aynı model ancak INT4 kuantizasyonu ($3.5\text{ GB} + 2\text{ GB tampon} = 5.5\text{ GB}$) ile bu donanıma sığdırılabilir.

---

### 2.2 KV-Cache (Key-Value Önbellek) Bellek Darboğazı

Otoregresif (Decoder-Only) modellerde üretilen her yeni kelime için geçmiş token'ların Key ve Value tensörleri bellekte saklanır. $L$ bağlam uzunluğu (context length), $n_{\text{layers}}$ katman sayısı, $n_{\text{heads}}$ dikkat başlığı sayısı, $d_{\text{head}}$ başlık boyutu, $b$ yığın boyutu (batch size) olmak üzere KV-Cache boyutu:

$$M_{\text{kv}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L \times b \times \text{Precision\_Bytes}$$

* **Örnek Hesaplama (Llama-2-7B):**  
  $n_{\text{layers}} = 32$, $n_{\text{heads}} = 32$, $d_{\text{head}} = 128$, FP16 (2 Bayt), $b = 1$ için:
  * **2.048 token bağlamda:** $\approx 1.07\text{ GB}$ ek VRAM
  * **8.192 token bağlamda:** $\approx 4.29\text{ GB}$ ek VRAM
  * **32.768 token bağlamda:** $\approx 17.18\text{ GB}$ ek VRAM

> **Sonuç:** Model ağırlıkları VRAM'e sığsa dahi, uzun bağlamlı promptlar verildiğinde KV-Cache VRAM'i taşırır ve sistemi çalışma zamanında kilitler.

---

### 2.3 Hesaplama Darboğazı: Memory-Bandwidth vs Compute-Bound

* **Ön Doldurma Aşaması (Prefill Phase):** Kullanıcının gönderdiği ilk prompt tek seferde matris çarpımıyla ($GEMM$) işlendiği için donanımın **hesaplama gücüne (TFLOPS - Compute Bound)** takılır.
* **Kod Çözme Aşaması (Decoding Phase):** Metin üretiminde her yeni token tek tek üretilir. Her token için modelin tüm milyarlarca ağırlığı VRAM'den çekirdeklere (Compute Cores) tekrar tekrar taşınmak zorundadır. Bu aşama tamamen **bellek bant genişliğine (Memory Bandwidth Bound)** bağımlıdır.
  * Standart RAM / PCIe 4.0: $\approx 32\text{ - }64\text{ GB/s}$ *(CPU'da LLM'in saniyede 1-2 kelime üretmesinin temel nedeni)*
  * GDDR6 (RTX 4090): $\approx 1.008\text{ GB/s}$
  * HBM3 (NVIDIA H100): $\approx 3.350\text{ GB/s}$ *(Endüstriyel hız)*

---

##  3. Platform Kurulum Matrisi (Yerelleştirilmiş Kılavuz)

### 3.1 Windows Mimarisi (NTFS ve CLI Optimizasyonları)

Hugging Face resmi dokümanı Windows'u kapsamaz. Windows üzerinde karşılaşılan kritik sistem engelleri ve mühendislik çözümleri:

1. **NTFS Sembolik Bağlantı (Symlink) Engeli:**
   Hugging Face Hub, modelleri önbelleğe alırken disk alanından tasarruf etmek için `symlinks` kullanır. Windows'ta varsayılan olarak bu yetki kapalıdır.
   * **Çözüm:** Windows Ayarlarından *Geliştirici Modu (Developer Mode)* aktif edilmelidir veya ortam değişkeni tanımlanmalıdır:
     ```powershell
     [System.Environment]::SetEnvironmentVariable('HF_HUB_DISABLE_SYMLINKS_WARNING', '1', 'User')
     ```
2. **PowerShell Betik Çalıştırma Politikası (ExecutionPolicy):**
   Sanal ortam aktivasyonunda `Activate.ps1 cannot be loaded` hatası için:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
3. **UTF-8 Bayt Akışı (cp1254 Karakter Kodlaması Çökmesi):**
   Konsol çıktılarında Türkçe ve Unicode karakterlerin Python betiğini `UnicodeEncodeError` ile düşürmemesi için betiklerin başına şu sistem katmanı eklenmiştir:
   ```python
   import sys
   if sys.stdout.encoding != 'utf-8':
       sys.stdout.reconfigure(encoding='utf-8')
   ```

---

### 3.2 Sanal Ortam (Virtual Environment) Yaşam Döngüsü

```bash
# 1. Proje kök dizininde izole ortam oluşturma
python -m venv nlp_env

# 2. Sanal ortamı aktif etme
# Windows PowerShell:
.\nlp_env\Scripts\Activate.ps1
# Linux / macOS:
source nlp_env/bin/activate

# 3. Paket yöneticisini ve bağımlılıkları yükleme
pip install --upgrade pip
pip install -r bolum_00_kurulum/requirements.txt
```

---

## 4. Adli Bilişim & Çalışma Zamanı (Runtime) Zafiyet Analizi

Bu bölümde Hugging Face ve PyTorch ekosisteminin çalışma zamanındaki 4 temel zafiyet noktası canlı testlerle belgelenmiştir. Bu deneylerin tamamı [`bolum_00_kurulum/kurulum_ve_zafiyet_analizi.ipynb`](kurulum_ve_zafiyet_analizi.ipynb) dosyasında çalıştırılabilir durumdadır.

---

### Deney 1: Sayısal Kararsızlık — FP16 Exponent Overflow & `NaN` Üretimi

* **Zafiyet Mekanizması:** Attention formülündeki $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ ifadesinde yer alan $\sqrt{d_k}$ ölçekleme faktörü unutulursa ne olur?
* **Matematiksel Açıklama:**  
  16-bit kayan nokta (FP16) formatında temsil edilebilecek **maksimum pozitif sayı $65.504$'tür**.  
  Softmax işlemi sırasında $e^x$ hesaplanır. $x \ge 11.09$ olduğunda $e^{11.09} > 65.504$ sınırını aşar. Logitler 70 civarına çıktığında $e^{70} = \infty$ (`inf`) olur.  
  Birden fazla sonsuz değerin bölümü $\frac{\infty}{\infty} = \text{NaN}$ (Not a Number) üretir ve modelin tüm dikkat matrisi anında çöker!

```python
import torch

# Sayısal taşma simülasyonu
logits = torch.tensor([70.0, 70.0], dtype=torch.float16)
# Naive Softmax: exp(x) / sum(exp(x))
overflow_exp = torch.exp(logits)
# exp([70, 70]) -> tensor([inf, inf], dtype=torch.float16)
nan_result = overflow_exp / torch.sum(overflow_exp)
print(nan_result) # Çıktı: tensor([nan, nan], dtype=torch.float16)
```

> **Mühendislik Bulgusu:** Vaswani vd. (2017) makalesindeki $\sqrt{d_k}$ bölme işlemi keyfi bir hiperparametre değil; FP16/BF16 donanımlarda Softmax'in sonsuza patlamasını (`inf/inf -> NaN`) engelleyen **hayati bir sayısal stabilizasyon kalkanıdır**.

---

### Deney 2: Donanım Cihaz Uyumsuzluğu (CUDA / CPU Device Mismatch)

* **Zafiyet Mekanizması:** PyTorch, PCIe veri yolu üzerinden tensörleri cihazlar arasında (Host RAM $\leftrightarrow$ GPU VRAM) otomatik olarak kopyalamaz (implicit copy yapmaz).
* **Oluşan Hata:**
  ```text
  RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
  ```
* **Kök Neden:** Model ağırlıkları GPU çekirdeklerinde (`cuda:0`) iken, girdi tensörü `model(inputs)` çağrısına CPU tensörü olarak iletilirse C++ kernel launch aşamasında tensör bellek işaretçileri uyuşmaz ve çalışma zamanı çöker.
* **Savunma Kodu:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  inputs = {k: v.to(device) for k, v in inputs.items()}
  ```

---

### Deney 3: Güvenlik Duvarı ve Kimlik Doğrulama Sınırı (Gated Repo 401)

* **Zafiyet Mekanizması:** Açık ağırlıklı modeller (örn: Meta Llama-3, Mistral) lisans onayına tabi "Gated Repository" statüsündedir. Token doğrulaması olmadan model çekilmeye çalışıldığında API seviyesinde erişim reddedilir.
* **Canlı Test Sonucu:**
  ```python
  from transformers import AutoTokenizer
  # Token verilmeden Llama-2'ye istek atıldığında:
  tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=False)
  ```
  **Dönen Hata:**
  ```text
  OSError: You are trying to access a gated repo.
  Make sure to have access to it at https://huggingface.co/meta-llama/Llama-2-7b-hf.
  401 Client Error: Unauthorized.
  ```

---

### Deney 4: Bellek Taşması ve OOM (Out-of-Memory) Dinamiği

* **Zafiyet Mekanizması:** PyTorch Caching Allocator, işletim sisteminden büyük bellek blokları rezerve eder. Model boyutu + Batch boyutu + Context uzunluğu VRAM sınırını 1 bayt dahi aştığında işletim sistemi seviyesinde kurtarma yapılamaz ve `torch.cuda.OutOfMemoryError` fırlatılır.
* **Kurtarma ve Önleme Stratejileri:**
  1. **Gradient Checkpointing:** Bellekteki ara aktivasyonları saklamak yerine geriye yayılımda yeniden hesaplayarak %60'a varan bellek tasarrufu sağlar.
  2. **FlashAttention:** Bellek erişim karmaşıklığını $O(N^2)$'den $O(N)$'e düşürerek SRAM üzerinde hesaplar.
  3. **Kuantizasyon (BitsAndBytes):** Modeli 8-bit veya 4-bit (NF4) formatında yüklemek.

---

## 5. Akademik Değerlendirme ve Sınav Soruları

Bu sorular, ünitenin yüzeysel ezberini değil, sistem ve donanım mimarisi seviyesindeki mantığını ölçmek üzere tasarlanmıştır.

### Soru 1:
7 Milyar ($7 \times 10^9$) parametreye sahip modern bir LLM'in ağırlıklarını **FP16 (16-bit Float)** hassasiyetinde GPU belleğine (VRAM) yüklemek isteyen bir sistem mühendisinin, KV-Cache ve aktivasyon tamponları **hariç**, yalnızca model ağırlıkları için ayırması gereken minimum VRAM miktarı nedir?

- A) 3.5 GB  
- B) 7.0 GB  
- **C) 14.0 GB (DOĞRU CEVAP)**  
- D) 28.0 GB  

> **Çözüm ve Çeldirici Analizi:**  
> FP16 hassasiyetinde her bir parametre $16\text{ bit} = 2\text{ Bayt}$ yer kaplar.  
> Toplam boyut: $7 \times 10^9 \times 2\text{ Bayt} = 14 \times 10^9\text{ Bayt} \approx 14\text{ GB}$.  
> *Çeldiriciler:* A şıkkı (3.5 GB) INT4 kuantizasyon sonucudur. B şıkkı (7 GB) INT8 kuantizasyon sonucudur. D şıkkı (28 GB) FP32 (32-bit float) hassasiyeti sonucudur.

---

### Soru 2:
Self-Attention mekanizmasındaki $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ formülünde, $\sqrt{d_k}$ ölçekleme teriminin kullanılmasının **donanımsal ve sayısal (numerical)** gerekçesi aşağıdakilerden hangisinde doğru açıklanmıştır?

- A) Dikkat matrisinin satır ve sütun boyutlarını eşit tutmak.  
- B) GPU çekirdeklerinin matris çarpımını daha az kayan nokta işlemiyle (FLOP) tamamlamasını sağlamak.  
- **C) $d_k$ boyutu büyüdükçe iç çarpım değerlerinin aşırı büyümesini önleyerek, Softmax fonksiyonunun türevinin sıfıra yaklaşmasını (vanishing gradient) ve FP16 formatında üstel taşma (exponent overflow / `NaN`) oluşmasını engellemek. (DOĞRU CEVAP)**  
- D) Pozisyonel kodlama (Positional Encoding) vektörlerinin sıralamasını korumak.

> **Çözüm ve Çeldirici Analizi:**  
> İki rastgele $d_k$ boyutlu ortalaması 0, varyansı 1 olan vektörün iç çarpımı $d_k$ varyansına sahiptir. $d_k$ büyüdükçe (örn. 64 veya 128) iç çarpım değerleri çok büyük pozitif/negatif sayılara ulaşır. $\sqrt{d_k}$ ile bölünmezse, Softmax girdileri aşırı büyür ve türevleri neredeyse 0 olur (öğrenme durur). Ayrıca FP16 hassasiyetinde $e^x$ hesaplanırken 65.504 üst sınırını aşarak `inf/inf = NaN` hatasına yol açar.

---

### Soru 3:
Bir dil modelinde çıkarım (inference) yapılırken, ilk istemin işlendiği **Prefill aşaması** ile metnin tek tek üretildiği **Decode aşaması** arasındaki donanımsal darboğaz farkı nedir?

- A) Her iki aşama da tamamen CPU bellek bant genişliğine bağımlıdır.  
- **B) Prefill aşaması tüm istemi matris-matris çarpımıyla tek seferde paralel işlediği için hesaplama gücüne (Compute-Bound / TFLOPS); Decode aşaması ise her kelimede tüm model ağırlıklarını bellekten tekrar tekrar çektiği için bellek bant genişliğine (Memory-Bandwidth Bound) bağımlıdır. (DOĞRU CEVAP)**  
- C) Prefill aşaması yalnızca disk hızına bağlıdır, Decode aşaması ise GPU VRAM hızına bağlıdır.  
- D) Decode aşamasında KV-Cache kullanılmadığı için hesaplama maliyeti her zaman sıfırdır.

---

## Dosya İndeksi ve Çıktılar

| Dosya Adı | Açıklama |
| :--- | :--- |
| [`bolum_00_kurulum/README.md`](README.md) | Bu ana dokümantasyon, donanım matematiği ve zafiyet raporu. |
| [`bolum_00_kurulum/requirements.txt`](requirements.txt) | Üretim standardında kütüphane ve paket bağımlılıkları. |
| [`bolum_00_kurulum/test_setup.py`](test_setup.py) | Donanım kontrolü, sayısal stabilite testi ve model kıyaslama betiği. |
| [`bolum_00_kurulum/kurulum_ve_zafiyet_analizi.ipynb`](kurulum_ve_zafiyet_analizi.ipynb) | 4 zafiyet deneyini interaktif olarak koşturan Jupyter Notebook. |
