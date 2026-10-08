# Week 2 — Keyword Relevance Score vs Week 1 Manual Labels

16 of 75 retrieved documents overlap with the Week 1 manually-labeled set (matched by url).

Manual labels: R = Relevant, P = Partially relevant, I = Irrelevant.
Score-implied label uses simple thresholds for illustration only (>=0.5 = 'R-like', 0.25-0.5 = 'P-like', <0.25 = 'I-like') - these thresholds are NOT the frozen Week 4 threshold, just a way to make disagreements visible now.

| url (truncated) | manual label | keyword score | score-implied | agree? |
|---|---|---|---|---|
| https://argumenti.ru/economics/zasuha-v-brazilii-mir-gotovit... | R | 0.000 | I | no |
| https://biz.heraldcorp.com/article/10820318... | R | 0.000 | I | no |
| https://vz.ru/news/2026/8/23/1445226.html... | R | 0.000 | I | no |
| https://esgnow.republika.co.id/berita/tgwvks416/el-nino-meng... | R | 0.016 | I | no |
| https://www.2merkato.com/news/trading/8952-global-coffee-pro... | P | 0.305 | P | yes |
| https://g1.globo.com/economia/agronegocios/noticia/2026/07/0... | R | 0.000 | I | no |
| https://g1.globo.com/es/espirito-santo/noticia/2026/07/12/te... | R | 0.000 | I | no |
| https://globorural.globo.com/previsao-do-tempo/noticia/2026/... | P | 0.000 | I | no |
| https://www.infomoney.com.br/business/chuvas-afetam-qualidad... | P | 0.000 | I | no |
| https://globorural.globo.com/cotacoes/noticia/2026/07/preco-... | R | 0.000 | I | no |
| https://jornaldebrasilia.com.br/noticias/economia/el-nino-am... | R | 0.000 | I | no |
| https://noticiabrasil.net.br/20260825/53486397.html... | P | 0.000 | I | no |
| https://www.elperiodico.com/es/economia/20260828/produccion-... | R | 0.005 | I | no |
| https://news.tuoitre.vn/robusta-coffee-climate-resilience-an... | P | 0.288 | P | yes |
| https://www.timesfreepress.com/news/2026/jul/18/free-press-o... | I | 0.000 | I | yes |
| https://jmonline.com.br/geral/seca-vira-maior-inimiga-do-caf... | R | 0.000 | I | no |

**Agreement rate: 3/16 (19%)**

**False positives (0)** — keyword score rated 'relevant-ish' but I manually labeled Irrelevant:
- (none in this overlap)

**False negatives (10)** — keyword score rated 'irrelevant-ish' but I manually labeled Relevant:
- [0.000] Засуха в Бразилии – мир готовится к дефициту кофе — https://argumenti.ru/economics/zasuha-v-brazilii-mir-gotovitsya-k-deficitu-kofe-1004132
- [0.000] 폭염에 커피 · 빵값 또 오르나 … 슈퍼 엘니뇨  농산물 시장 강타 [ 나우 , 어스 ]  — https://biz.heraldcorp.com/article/10820318
- [0.000] Bloomberg : Тропические дожди в Бразилии ударили по мировому рынку арабики :: Новости дня / ВЗГЛЯД — https://vz.ru/news/2026/8/23/1445226.html
- [0.016] El Nino Menguat , Pasokan Kopi Global Terancam — https://esgnow.republika.co.id/berita/tgwvks416/el-nino-menguat-pasokan-kopi-global-terancam
- [0.000] Do café ao arroz : El Niño ameaça produo e pode elevar preços dos alimentos — https://g1.globo.com/economia/agronegocios/noticia/2026/07/08/do-cafe-ao-arroz-el-nino-ameaca-producao-e-pode-elevar-precos-dos-alimentos.ghtml
- [0.000] Temor de possível super El Niño faz preço do café disparar — https://g1.globo.com/es/espirito-santo/noticia/2026/07/12/temor-de-super-el-nino-faz-preco-do-cafe-disparar-e-alta-pode-pesar-no-bolso-dos-consumidores.ghtml
- [0.000] Preço do café sobe em Nova York com mercado atento ao El Niño — https://globorural.globo.com/cotacoes/noticia/2026/07/preco-do-cafe-sobe-em-nova-york-com-mercado-atento-ao-el-nino.ghtml
- [0.000] El Niño ameaça safras de cacau , café e acar em países tropicais — https://jornaldebrasilia.com.br/noticias/economia/el-nino-ameaca-safras-de-cacau-cafe-e-acucar-em-paises-tropicais/
- [0.005] La producción mundial de café se tambalea tras el terremoto de Colombia y la llegada del fenómeno climático de El Niño — https://www.elperiodico.com/es/economia/20260828/produccion-mundial-cafe-tambalea-terremoto-colombia-fenomeno-nino-133744749
- [0.000] Seca vira maior inimiga do café e ajuda a manter preço alto — https://jmonline.com.br/geral/seca-vira-maior-inimiga-do-cafe-e-ajuda-a-manter-preco-alto-1.652734

## Interpretation

A basic keyword/vocabulary overlap score is expected to disagree with human judgment in specific, explainable ways - not randomly. False positives are the more dangerous failure mode for this project (per the relevance-gating principle: an over-eager score feeding a confident-but-wrong LLM explanation), so they matter more here than false negatives. Any false positives above should be inspected for *why* they scored high - the most likely cause is surface keyword overlap without the disruption-vocabulary term appearing in the true causal sense (e.g. a broker product page mentioning 'Robusta' or a festival page mentioning 'Brazil' and 'coffee'). This is exactly the failure mode Week 6's relevance gate needs to catch, and this comparison is preliminary evidence for why keyword-only scoring alone (without semantic retrieval or a proper gate) is not sufficient.