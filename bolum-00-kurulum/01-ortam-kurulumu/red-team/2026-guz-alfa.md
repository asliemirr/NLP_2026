# Bolum 0: Kurulum ve Calisma Zamani Hata Analizi (Red-Team)

**Ders:** Dogal Dil Isleme (NLP)  
**Ogretim Uyesi:** Dr. Mesut Polatgil  
**Takim:** Alfa (Mehmet Can Efe)  
**Konu:** Kurulum Asamasinda Karsilasilan Hatalar ve Cozumleri  

---

## 1. Hata 1: Sanal Ortam Aktif Edilmeden Kod Calistirma

* **Hata:** `ModuleNotFoundError: No module named 'transformers'`
* **Neden:** Kutuphaneler sanal ortama kurulmus ancak calistirma sirasinda sanal ortam aktif edilmemistir.
* **Cozum:** Sanal ortam aktiflestirilir:
  * Windows: `.\.env\Scripts\Activate.ps1`
  * Linux/Mac: `source .env/bin/activate`

---

## 2. Hata 2: PowerShell Calistirma Politikasi Engeli

* **Hata:** `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
* **Neden:** Windows PowerShell'in varsayilan guvenlik politikasi betik calistirmaya izin vermez.
* **Cozum:** PowerShell yetkisi guncellenir:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

---

## 3. Hata 3: Token Olmadan Ozel (Gated) Modele Erisim

* **Hata:** `OSError: 401 Client Error: Unauthorized for url...`
* **Neden:** Llama veya ozel onay gerektiren depolara tokensiz istek atilmistir.
* **Cozum:** `huggingface-cli login` calistirilarak yetkili hesap token'i girilmelidir.
