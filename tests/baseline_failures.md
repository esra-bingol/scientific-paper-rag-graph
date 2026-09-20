# Naive RAG baseline başarısızlıkları

Tarih: 2026-09-20  
Sistem: `src/05_query.py` — Chroma `papers_naive`, top-5, `gpt-4o-mini`  
Külliyat: 20 makale, 1818 naive chunk  
`recall@5` / `faithfulness`: gold set yok; aşağıdakiler elle gözlem.

Amaç: vektör-k-NN’nin **yapısal olarak** çözemediği soru tiplerini kilitlemek. Sonraki aşamalar (metadata filter, OpenAlex `author_id`, GROBID atıf grafı, RRF) bunları hedef alır.

Cosine **uzaklık** (küçük = daha yakın). Bu koşularda 0.50+ zayıf eşleşme.

---

## A. Tarih filtresi

Gold yıl `data/metadata.json` içinden (chunk metadata’da `year` yok).

### 1. 2023’ten sonra yayımlanan makalelerin arXiv id’lerini listele.

- **Naive RAG’de ne oldu:** Top-5: `2310.20476v1` (2023), `2404.14462v4`, `2405.12573v1` (uzaklık 0.62–0.69). Cevap yalnızca `2404.14462v4` ve `2405.12573v1` dedi. Gold (yıl > 2023): yedi id (`2404.14462`, `2405.12573`, `2411.07602`, `2502.06243`, `2502.13721`, `2507.13354`, `2603.21376`).
- **Neden başarısız:** Cosine “2023’ten sonra”yı sayısal filtre gibi uygulamaz; top-5 külliyatın tamamını sayamaz.
- **Hangi aşamada çözülür:** Ingest’teki `year` alanını chunk metadata’ya yazıp Chroma `where` (veya SQL) ile yıl kesmek.

### 2. 2020’den önce çıkmış külliyattaki Transformer makalelerinin başlıkları nedir?

- **Naive RAG’de ne oldu:** Top-5’in dördü 2023–2024 (`2404`, `2305`, `2310`); doğru kağıt `1910.13634v1` (2019) 5. sırada (0.58). Cevap külliyat başlığını vermedi; chunk içindeki **kaynakça** maddelerini uydurur gibi listeledi (Nguyen 2019, Shazeer 2020).
- **Neden başarısız:** Yıl semantik benzerlik değil; naive extract kaynakçayı gövdeyle karıştırdığı için LLM alıntı başlığını “bizim makale” sandı.
- **Hangi aşamada çözülür:** `year < 2020` metadata filter + GROBID `listBibl` ayrımı (kaynakça chunk’ları indeksten veya en azından cevaptan düşer).

### 3. Sadece 2025 tarihli makalelerde kuantum veya fiziksel Transformer modeli var mı?

- **Naive RAG’de ne oldu:** Doğru kağıt `2507.13354v2` 1. ve 4.–5. sırada geldi; araya 2023 (`2305.18475v4`) ve 2024 (`2404.14462v4`) girdi. Cevap evet dedi ama külliyatta olmayan “topos-theoretic quantum AI” başlığını da ekledi (muhtemel atıf sızıntısı).
- **Neden başarısız:** Konu vektörü yılı ezer; “sadece 2025” kesilmediği için 2023–2024 parçaları prompt’u kirletir.
- **Hangi aşamada çözülür:** Önce `year=2025` filter, sonra aynı soruyla dense retrieval.

---

## B. Yazar filtresi

Gold yazar listesi metadata’da; kanonik kimlik henüz yok (`OpenAlex author_id` kuralı ileride).

### 4. Esra Bingöl’ün bu koleksiyondaki makaleleri hangileri?

- **Naive RAG’de ne oldu:** Top-5 tamamen alakasız (uzaklık 0.78–0.81: `2603.21376v1`, `2103.04037v2`, …). Cevap: “Bu bilgi verilen makalelerde yok.” (içerik olarak doğru abstain.)
- **Neden başarısız:** Yazar indeksi yok; sistem yine 5 chunk çekip prompt doldurdu — doğru cevap prompt kuralına bağlı şans, filtre değil.
- **Hangi aşamada çözülür:** Yazar metadata / OpenAlex `author_id` ile eşleşen sıfır kayıt → retrieval’sız “yok”.

### 5. Li soyadlı yazarların tüm makalelerini listele.

- **Naive RAG’de ne oldu:** Top-5 neredeyse hepsi `2603.21376v2` (Li yok), bir `2305.18475v4` chunk. Cevap: yok. Gold: Hailiang Li + Wenye Li → `1910.13634v1`; Xiaoyu Li → `2411.07602v2`; Qianxiao Li → `2305.18475v4`.
- **Neden başarısız:** Soyadı vektör uzayında kimlik değil; aynı “Li” birden fazla kişidir ve chunk’ta geçmeyince kağıt hiç gelmez.
- **Hangi aşamada çözülür:** Entity resolution (OpenAlex `author_id` kanonik) + yazar→makale metadata join.

### 6. Zhao Song hangi yıl hangi makaleyi yayımlamış?

- **Naive RAG’de ne oldu:** 4/5 chunk doğru kağıttan (`2411.07602v2`, 2024, RoPE complexity). Cevap ise kaynakçadaki **başka** Zhao Song makalelerini uydurdu (learning rate 2023, gradient complexity 2024); külliyat başlığını söylemedi.
- **Neden başarısız:** İsim hem yazar satırında hem referans listesinde geçer; naive RAG “bu kağıdın yazarı” ile “alıntılanan isim”i ayırmaz.
- **Hangi aşamada çözülür:** GROBID `author` vs `listBibl` ayrımı + OpenAlex `author_id` ile kanonikleştirme.

---

## C. İlişki (“X’i eleştiren…”)

Atıf / karşıtlık kenarı yok; sadece cosine.

### 7. Attention is All You Need’i (Vaswani et al. 2017) eleştiren veya sınırlarını kanıtlayan makaleler hangileri?

- **Naive RAG’de ne oldu:** Top-5: lottery ticket (`2005.03454v2`), kuantum (`2507.13354v2`), time-series (`2502.13721v1`), indoor (`2310.20476v1`), jest (`2211.02643v1`) — uzaklık 0.57–0.63. Cevap: yok. Aday (sınır kanıtı): `2411.07602v2` (RoPE circuit complexity) retrieval’a girmedi.
- **Neden başarısız:** “Eleştirir / sınırını kanıtlar” bir **kenar tipi**; Vaswani geçen her chunk bu ilişkiyi taşımaz.
- **Hangi aşamada çözülür:** GROBID atıflarından citation graph + (gerekirse) atıf cümlesi; graph retrieval ve RRF (vektör + graf).

### 8. Vanilla Transformer’ı indoor oda sıcaklığı tahmininde yetersiz bulup değiştiren çalışma hangisi?

- **Naive RAG’de ne oldu:** Top-5’in beşi doğru kağıt `2310.20476v1` (uzaklık 0.41–0.50, bu koşunun en sıkı eşleşmesi). Cevap yine “yok.”
- **Neden başarısız:** Konu eşleşti ama “yetersiz bulup değiştirme” iddiası naive 512’lik dilimde yok / model ilişkiyi çıkarmayı reddetti.
- **Hangi aşamada çözülür:** GROBID bölüm + atıf bağlamı (Related Work’te vanilla ile kıyas) ve/veya graph kenarı `improves_on`; tek cosine yetmez.

---

## D. Multi-hop (“X’in yazdığı Y yöntemi…”)

İki varlık birleştirilmeli; tek vektör hop’u yetmez.

### 9. Hailiang Li’nin yazdığı sinüzoidal positional encoding iyileştirmesi, Jan Steckel’in EchoPT modelinde kullanılmış mı?

- **Naive RAG’de ne oldu:** 4/5 Hailiang kağıdı (`1910.13634v1`), 1 indoor/RoPE (`2310.20476v1_0019`). EchoPT (`2405.12573v1`, Steckel) hiç gelmedi. Cevap: yok. (Muhtemel doğru abstain, ama EchoPT tarafı hiç okunmadı.)
- **Neden başarısız:** Sorgu tek vektör; iki yazar+yöntem+kullanım üç hop’u aynı anda çekemez.
- **Hangi aşamada çözülür:** Graph hop: `author—writes→paper—has_method→PE` ve `author—writes→EchoPT`, sonra atıf/kullanım kenarı veya ikinci retrieval; birleştirme RRF.

### 10. Zhao Song’un yazdığı RoPE karmaşıklık sonucunun, Alfredo V Clemente’nin indoor Transformer çalışmasına etkisi nedir?

- **Naive RAG’de ne oldu:** Top-5 yalnızca Song kağıdı `2411.07602v2` (0.53–0.58). Clemente `2310.20476v1` sıfır. Cevap: yok.
- **Neden başarısız:** Embedding sorunun bir ucuna (RoPE/Song) yapışır; ikinci kağıda ve aradaki “etki” kenarına atlamaz.
- **Hangi aşamada çözülür:** Çok hop’lu graph traversal (yazar→makale→atıf/yıl) + gerekirse iki ayrı retrieve’in RRF füzyonu; naive weighted sum yok.

---

## Özet

| Kategori | Soru | Retrieval | Cevap | Kök neden |
| --- | --- | --- | --- | --- |
| Tarih | 1 | 2023 kağıdı da geldi, 7 id’den 2’si | eksik liste | yıl ≠ cosine |
| Tarih | 2 | 2019 kağıdı 5. sıra | kaynakça halüsinasyonu | yıl + listBibl |
| Tarih | 3 | doğru 2025 + yanlış yıllar | evet + yabancı başlık | filter yok |
| Yazar | 4 | uzaklık ~0.80 gürültü | abstain (şanslı) | yazar indeksi yok |
| Yazar | 5 | Li kağıtları yok | yanlış “yok” | soyadı ≠ entity |
| Yazar | 6 | doğru kağıt var | yanlış başlıklar | yazar vs referans |
| İlişki | 7 | konu dağınık | yanlış “yok” | kenar tipi yok |
| İlişki | 8 | doğru kağıt | yanlış “yok” | ilişki cümlesi yok |
| Multi-hop | 9 | yalnız Li; EchoPT yok | doğrulanmamış “yok” | tek hop |
| Multi-hop | 10 | yalnız Song; Clemente yok | “yok” | tek hop |

On sorunun onunda dense top-5 ya yanlış küme getirdi, ya doğru kümeyi **filtre/ilişki/hop olmadan** getirdi. Cevabın bazen “yok” demesi başarı değil; mekanizma metadata ve grafa ihtiyaç duyuyor.
