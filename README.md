# scientific-paper-rag-graph

Bilimsel makaleler üzerinde **RAG + bilgi grafı** öğrenme projesi.

Amaç: PDF'ten metin çıkarıp parçalamak, vektör araması yapmak, yazar/atıf grafıyla birleştirmek ve cevabı kaynaklarıyla üretmek. RAG'a yeni başlayan biri için 3–4 haftalık, aşama aşama ilerleyen bir iskelet.

Bu repoda henüz pipeline kodu yok; önce klasörler ve kurulum var. Scriptler `01_*.py`, `02_*.py` sırasıyla eklenecek. Aşama atlanmaz.

## Ne öğreneceğiz

| Konu | Neden |
| --- | --- |
| GROBID | PDF'i düz metin değil; başlık, özet, bölüm, referans olarak yapılandırır. |
| Chunking stratejileri | Yanlış parça boyutu retrieval'ı bozar; karşılaştırarak seçeriz. |
| Entity resolution | Aynı yazar farklı yazılışlarla gelir; kanonik kimlik OpenAlex `author_id` olur. |
| RRF (Reciprocal Rank Fusion) | Vektör + keyword (veya graph) sıralarını birleştirir; naive weighted sum kullanmayız. |
| Evaluation | Her aşamada `recall@k` ve `faithfulness` raporlanır; "çalışıyor gibi" yetmez. |
| Cache (`diskcache`) | Aynı retrieval/LLM çağrısı tekrar edilmez; para ve zaman tasarrufu. |

## Klasör yapısı

```
data/raw_pdfs/          # indirilen PDF'ler
data/processed_text/    # GROBID / PyMuPDF çıktısı
data/chunks/            # chunking sonuçları
data/graph/             # yazar-atıf grafı
data/eval/              # soru-cevap ve metrik tabloları
src/                    # 01_*.py, 02_*.py scriptleri
notebooks/              # keşif not defterleri
tests/                  # birim testleri
docs/                   # mimari ve ders notları
cache/                  # diskcache önbelleği (git'e girmez)
```

`data/` ve `cache/` `.gitignore` ile dışarıda tutulur.

## Gereksinimler

- Python 3.11 veya üzeri
- OpenAI API anahtarı (embedding + LLM)

## Kurulum

```bash
# 1. Sanal ortam
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Bağımlılıklar
pip install -r requirements.txt

# 3. API anahtarı
cp .env.example .env
# .env içinde OPENAI_API_KEY=... satırını doldur
```

Anahtar `python-dotenv` ile okunur; koda yazılmaz.

## Çalıştırma

Henüz script yok. Eklendikçe her dosyanın başındaki docstring "nasıl çalıştırılır"ı yazar. Örnek:

```bash
python src/01_ornek.py
```

Loglar `print()` ile gelir; bu kasıtlı, öğrenme için.

## Değerlendirme kuralı

Her retrieval/LLM aşamasında en az şunlar yazdırılır:

- `recall@k`
- `faithfulness`

Fusion yapılırsa **RRF** kullanılır.

## Lisans ve veri

arXiv PDF'leri bu repoya commit edilmez. Kendi indirdiğin dosyalar `data/raw_pdfs/` altındadır.
