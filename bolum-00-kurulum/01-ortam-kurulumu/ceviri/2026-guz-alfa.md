> **Uyarı:** Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

# Bolum 0: Kurulum (Setup)

**Kaynak:** [Hugging Face LLM Course - Chapter 0 (Setup)](https://huggingface.co/learn/llm-course/chapter0/1)  
**Takim:** Alfa (Mehmet Can Efe)  

---

## Giris

Hugging Face kursuna hos geldiniz! Bu giris bolumu, calisma ortaminizi kurmanizda size rehberlik edecektir.

Bu kursta kullanacagimiz tum kutuphaneler Python paketleri olarak mevcuttur. Burada bir Python ortaminin nasil kurulacagini ve kutuphanelerin nasil yuklenecegini goreceksiniz.

Calisma ortaminizi kurmak icin iki temel yol bulunmaktadir:
1. Google Colab not defteri kullanmak
2. Python sanal ortami (virtual environment) kullanmak

Hugging Face hesabi olusturmaniz onerilir: [huggingface.co/join](https://huggingface.co/join)

---

## 1. Google Colab Not Defteri Kullanimi

Google Colab kullanmak en basit kurulum yoludur. Tarayicinizda bir not defteri acar ve dogrudan kodlamaya baslarsiniz.

1. Colab'da yeni bir not defteri acin.
2. Transformers kutuphanesini yukleyin:
   ```python
   !pip install "transformers[sentencepiece]"
   ```
3. Kurulumu test edin:
   ```python
   import transformers
   ```

---

## 2. Python Sanal Ortami (Virtual Environment) Kullanimi

Yerel bilgisayarinizda calismak istiyorsaniz, paketlerin sistem genelinde cakismamasi icin izole bir sanal ortam olusturmalisiniz.

### 2.1 Python Kontrolu
Terminalde Python surumunuzu kontrol edin:
```bash
python --version
```

### 2.2 Sanal Ortam Olusturma
Proje klasorunuzde sanal ortami olusturun:
```bash
python -m venv .env
```

### 2.3 Sanal Ortami Aktiflestirme
* **Windows (PowerShell):**
  ```powershell
  .\.env\Scripts\Activate.ps1
  ```
* **Windows (CMD):**
  ```cmd
  .\.env\Scripts\activate.bat
  ```
* **Linux / macOS:**
  ```bash
  source .env/bin/activate
  ```

---

## 3. Bagimliliklarin Yuklenmesi

Sanal ortam aktifken Transformers kutuphanesini kurun:
```bash
pip install "transformers[sentencepiece]"
```

---

## 4. Hugging Face Girisi

Terminalden Hugging Face hesabiniza giris yapin:
```bash
huggingface-cli login
```
Token soruldugunda [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresinden aldiginiz User Access Token'i yapistirin.
