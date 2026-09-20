# Ders notları

Her denemeden sonra bir satır ekle. Kod değişince burası da güncellenir.

Şablon (kopyala-yapıştır):

```
## YYYY-MM-DD — kısa başlık

- Ne denedim:
- Beklediğim:
- Gördüğüm:
- Neden (hipotez):
- recall@k:
- faithfulness:
- Sonraki adım:
```

---

## Kayıtlar

## 2026-09-20 — PyMuPDF düz metin: sütun ve referans kalitesi

İncelenen dosya: `data/processed_text/1910.13634v1_raw.txt`
(IEEE iki sütun: *An Augmented Transformer Architecture for NLG Tasks*)
Kontrol için bakılan ikinci dosya: `2103.04037v2_raw.txt` (Springer iki sütun)

- Ne denedim: `page.get_text()` ile tüm sayfaları uç uca eklemek (layout yok).
- Beklediğim: okunabilir gövde; sütunlar karışmasın; kaynakça ayrı kalsın.
- Gördüğüm:
  1. **İki sütun “fermuar” gibi satır satır karışmamış.** Sol sütun paragrafı çoğunlukla sırayla geliyor.
  2. **Layout yine bozulmuş.** Dar sütun satır sonları kelimeyi kesiyor (`Although` / `the` / `main` ayrı satırlar; `fun-` / `damental`).
  3. **Şekil, grafik ve yan kutu gövdeye giriyor.** `1910`: Fig.1 etiketleri (`Transformer Encoder`, `Softmax`) Related Work’ün ortasında; Fig.5–8 eksen sayıları Conclusions’tan sonra, REFERENCES’ten önce. `2103`: yazar e-postaları Introduction paragrafının içine düşüyor; `arXiv:...` ve `Andrew Shin et al.` koşu başlığı sayfa ortasında.
  4. **Kaynakça listesi gövdeye yapışmamış.** `REFERENCES` / `References` başlığından sonra başlıyor. Ama gövdedeki `[30]` atıfları kalıyor (bu doğru). Kaynakça bloğunun **içine** sayfa başlığı/numarası karışıyor (`2103` satır 2500–2503).
- Neden (hipotez): `get_text()` kutuları yaklaşık okuma sırasıyla dizer; iki sütunu “anlamaz”, şekil/header/footer’ı da metin sanır. GROBID bunları `div` / `figure` / `listBibl` diye ayırır.
- recall@k: n/a
- faithfulness: n/a (henüz retrieval/LLM yok)
- Sonraki adım: aynı PDF’leri GROBID ile çıkar; gövde vs referans ayrımını ve sütun birleştirmeyi karşılaştır.
