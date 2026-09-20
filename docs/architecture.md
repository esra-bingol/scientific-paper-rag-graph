# Mimari

Bu dosya şablondur. Her aşama bitince ilgili bölüm doldurulur. Aşama atlama.

## 1. Amaç

- Soru:
- Girdi:
- Çıktı:

## 2. Bileşenler

| Bileşen | Görev | Neden bu kütüphane? | Durum |
| --- | --- | --- | --- |
| PDF alma | | arxiv: makale indirme | boş |
| Metin çıkarma | | pymupdf / GROBID | boş |
| Chunking | | | boş |
| Embedding + vektör DB | | openai + chromadb | boş |
| Keyword / graph retrieval | | networkx | boş |
| Fusion | | RRF (weighted sum yok) | boş |
| LLM cevap | | openai | boş |
| Cache | | diskcache | boş |
| Entity resolution | | rapidfuzz; kanonik id = OpenAlex author_id | boş |
| Evaluation | | pandas; recall@k, faithfulness | boş |

## 3. Veri akışı

```
PDF (data/raw_pdfs/)
  -> processed text (data/processed_text/)
  -> chunks (data/chunks/)
  -> vektör indeks + graf (data/graph/)
  -> retrieval (+ cache/)
  -> RRF fusion
  -> LLM
  -> eval raporu (data/eval/)
```

Notlar:

-

## 4. Retrieval

- Dense (embedding):
- Sparse / keyword:
- Graph hop:
- Fusion: RRF (k parametresi: ___)

## 5. Graph

- Düğüm tipleri:
- Kenar tipleri:
- Entity resolution kuralı: OpenAlex `author_id` kanoniktir.

## 6. Cache

- Neler cache'lenir: retrieval sonuçları, LLM çağrıları
- Anahtar nasıl üretilir:
- Konum: `cache/`

## 7. Evaluation

Her aşamada raporla:

| Aşama | recall@k | faithfulness | Not |
| --- | --- | --- | --- |
| | | | |

## 8. Kararlar (neden)

- Karar:
- Alternatif:
- Neden bunu seçtik:

## 9. Riskler

-
