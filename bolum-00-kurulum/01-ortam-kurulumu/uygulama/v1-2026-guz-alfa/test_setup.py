# Dogal Dil Isleme - Bolum 0 Hizli Test Betigi
# Takim: Alfa (Mehmet Can Efe)

import sys

def test():
    print("[*] Bolum 0 Kurulum Testi Baslatiliyor...")
    print(f"[*] Python: {sys.version.split()[0]}")
    
    try:
        import torch
        import transformers
        from transformers import pipeline
        print(f"[+] PyTorch: {torch.__version__}")
        print(f"[+] Transformers: {transformers.__version__}")
        
        clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        res = clf("Kurulum basarili")[0]
        print(f"[+] Model Test Ciktisi: {res['label']} ({res['score']:.4f})")
        print("[OK] Kurulum testi basariyla tamamlandi.")
    except Exception as e:
        print(f"[X] Hata: {e}")

if __name__ == "__main__":
    test()
