# Doğal Dil İşleme (NLP) - Açık Kaynak Sistem Mimarisi

Bu depo, 4. sınıf Doğal Dil İşleme dersinin laboratuvar merkezi, yerelleştirilmiş dokümantasyon arşivi ve sızma testi (red-teaming) alanıdır. Dersin ana omurgası [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) üzerinden yürütülmektedir.

## 📌 İş Akışı: Fork ve Pull Request (PR) Kuralları

Orijinal depoyu (repo) temiz tutmak adına hiçbir öğrenciye doğrudan dal (branch) açma yetkisi (Collaborator) verilmemiştir. Tüm süreç sektör standardı olan **Fork (Çatallama)** yöntemiyle işleyecektir:

1. **Fork (Çatallama):** Takımın Dokümantasyon Yöneticisi (Öğrenci 1), orijinal repoya girip sağ üstteki "Fork" butonuna basarak deponun birebir kopyasını kendi GitHub hesabına alır.
2. **Geliştirme (.md / .qmd / .ipynb):** Takımın tüm üyeleri orijinal metnin çevirilerini, yerelleştirilmiş "Cheat Sheet" özetlerini ve Python hata ayıklama (forensics) kodlarını bu *kopya* repoda hazırlar. Word belgesi veya zip dosyası kesinlikle kabul edilmez.
3. **Lokal Test:** Kodlarınızı ve dokümanınızı göndermeden önce mutlaka kendi bilgisayarınızda derleyerek test edin.
4. **Onay İste (Pull Request):** İşiniz bittiğinde, Öğrenci 1 kendi kopya reposundan orijinal deponun `main` dalına dışarıdan bir PR açar. Hata veren veya tahtada mimari olarak savunulamayan kodlar PR aşamasında doğrudan reddedilecektir.

## 👥 Takım Rolleri ve Depo Katkısı

Sınıftaki 4 kişilik Yapay Zeka Ar-Ge Takımları, bu repoyu aşağıdaki görev dağılımına göre besleyecektir:

* **Öğrenci 1 (Dokümantasyon Yöneticisi):** Hugging Face orijinal metnini çevirir ve kendi anladığı yerelleştirilmiş teknik rehber ile birlikte PR atar.
* **Öğrenci 2 (Sistem Mimarı):** Repodaki dokümanı kullanarak algoritmanın temelini ve icat edilme nedenini tahtada sınıfa sezgisel olarak açıklar.
* **Öğrenci 3 (QA & Adli Bilişim Uzmanı):** Repodaki kodu kasten bozarak veya modelin sınırlarını zorlayarak sistemin zafiyetini (hallucination, prompt injection) tahtada canlı olarak test eder.
* **Öğrenci 4 (Sınav Komiseri):** Ünite sonu quizlerini akademik değerlendirme formatına getirerek bölüm dokümanının sonuna ekler ve sınıfı test eder.

## 📂 Klasör Mimarisi

```text
/bolum_01_transformer_modellerine_giris/
/bolum_02_huggingface_kutuphaneleri/
/bolum_03_ince_ayar_finetuning/
README.md
