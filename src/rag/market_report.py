"""
Which documents the blind reading should be shown first, and which part of
each.

The reading step is shown a dozen recent documents. Until pipeline version 9
they were the dozen that scored highest against one fixed phrase about supply
shocks, and each was shown from its first character. On the first run over
the labelled dates 133 of the 211 documents read said nothing about a price
move (a city guide to coffee shops, a product launch), a Reuters report headed
"Coffee prices jump nearly 7% in New York after stocks drawn" was retrieved
and left unread, and market reports that open with sugar or cocoa were cut
off before coffee was mentioned.

Three things here, all of them blind to the direction of the move:

  market_report_score   how much a headline looks like a report on coffee's
                        market price (it names a price or the market, it has
                        a verb of movement, it gives a percentage)
  focus_text            the title-led opening of a document followed by the
                        passages that mention coffee, so that a fixed-length
                        excerpt holds the part that matters
  session_date          the trading day a document can be reporting on: one
                        seen before the exchange opened belongs to the day
                        before

The word lists cover the languages the retrieved documents were actually in
(English, Portuguese, Spanish, Vietnamese, French, German, Italian,
Indonesian, Turkish, Russian, Chinese, Japanese, and from pipeline version
10 Arabic and Korean, added after a report of the day's move went unread on a
labelled date because its headline was in Arabic). They list verbs of rising
and of falling side by side and the score never says which was matched, so
it cannot favour one direction.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

# Coffee, in the languages above. Used to find the passages of a document
# that are about coffee.
COFFEE = re.compile(
    r"coffee|caf[eé]|kaffee|caff[eè]|c[aà] ph[eê]|kopi|kahve|кофе|咖啡|コーヒー|"
    r"قهوة|커피|원두|"
    r"ar[aá]bica|robusta|conilon",
    re.IGNORECASE)

# Words that place a headline on the commodity exchange: the contract, the
# exchange, the settlement, the variety that is traded.
_EXCHANGE = re.compile(
    r"\bfutures?\b|\bsofts\b|\bsettle[sd]?\b|\bice\b|\bcontracts?\b|\bnybot\b|"
    r"ar[aá]bica|robusta|conilon|"
    r"\bcota[cç][aã]o\b|\bcota[cç][oõ]es\b|\bcotizaci[oó]n(?:es)?\b|\bbolsa\b|\bfuturos?\b|"
    r"\bnova york\b|\bnueva york\b|\bnew york\b|\bny\b|\blondres\b|"
    r"期货|期貨|先物|фьючерс\w*|"
    r"العقود الآجلة|عقود آجلة|بورصة|أرابيكا|روبوستا|선물|아라비카|로부스타|\bvadeli\b|\bborsa\w*",
    re.IGNORECASE)

# A price, said generally.
_PRICE = re.compile(
    r"\bprices?\b|\bmarkets?\b|\bpre[cç]os?\b|\bprecios?\b|\bmercados?\b|"
    r"\bprix\b|\bcours\b|\bpreise?\b|\bkaffeepreise?\b|\bprezz[io]\b|\bgi[aá]\b|"
    r"\bharga\b|\bfiyat\w*|цен\w*|价格|價格|相場|価格|cijen\w*|"
    r"أسعار|سعر|가격|시세",
    re.IGNORECASE)

# Verbs a market report uses for a move, either way: the report's own
# vocabulary ("jumps", "tumbles", "settles", "dispara", "despenca").
_MOVE_REPORT = re.compile(
    r"\bjump\w*|\bsurg\w+|\bsoar\w*|\brall(?:y|ies|ied)\b|\bclimb\w*|\bspike\w*|\bplung\w+|"
    r"\bplummet\w*|\btumbl\w+|\bslump\w*|\bslid(?:e|es)?\b|\bsink\w*|\bslip\w*|\bretreat\w*|"
    r"\bsettle[sd]?\b|\bdrop(?:s|ped)?\b|\bfall(?:s)?\b|\bfell\b|\brise[s]?\b|\brose\b|"
    r"\bsobe[m]?\b|\bdispara[m]?\b|\bavan[cç]a[m]?\b|\bsalta[m]?\b|\bcai\b|\bcaem\b|\brecua[m]?\b|"
    r"\bdesaba[m]?\b|\bdespenca[m]?\b|\bfecha[m]?\b|\bencerra[m]?\b|\bopera[m]?\b|"
    r"\bsube[n]?\b|\bcae[n]?\b|\bbaja[n]?\b|\bcierra[n]?\b|"
    r"t[aă]ng m[aạ]nh|gi[aả]m m[aạ]nh|[đd][oồ]ng lo[aạ]t|\bgrimpe\w*|\bchute\w*|\bbondi\w*|"
    r"\bsteigt\b|\bf[aä]llt\b|\bsinkt\b|\bklettert\b|\bmelonjak\b|\banjlok\b|"
    r"вырос\w*|упал\w*|上涨|上漲|下跌|大涨|大跌|急騰|急落|"
    r"ترتفع|يرتفع|ارتفع|تقفز|قفز|تنخفض|ينخفض|انخفض|تهبط|هبط|تتراجع|تراجع|"
    r"급등|급락|폭등|폭락|상승|하락|"
    r"\by[uü]kseldi\b|\by[uü]kseliyor\b|\bd[uü][sş]t[uü]\b|\bd[uü][sş][uü]yor\b|\bgeriledi\b",
    re.IGNORECASE)

# Other words of movement or level, either way.
_MOVE = re.compile(
    r"\bris(?:e|es|ing)\b|\bgains?\b|\bhigher\b|\bhighs?\b|\bpeaks?\b|\brecord\b|\bfalling\b|"
    r"\bdrop\w*|\blower\b|\blows?\b|\beas(?:e|es|ed)\b|\bup\b|\bdown\b|"
    r"\balta[s]?\b|\bvaloriza\w*|\bm[aá]xima[s]?\b|\bm[ií]nima[s]?\b|\bpico\b|\bqueda[s]?\b|"
    r"\bbaixa[s]?\b|\bdesvaloriza\w*|\bperdas?\b|\bganhos?\b|"
    r"\balza[s]?\b|\bca[ií]da[s]?\b|t[aă]ng|gi[aả]m|\bhausse\b|\bbaisse\b|\bflamb\w+|"
    r"\bsteig\w+|\bf[aä]ll\w+|\bnaik\b|\bturun\b|"
    r"рост\w*|паден\w*|подорожа\w*|подешев\w*|新高|新低|高値|安値|"
    r"ارتفاع|صعود|انخفاض|هبوط|인상|인하|최고|최저|"
    r"\by[uü]kseli[sş]\b|\bd[uü][sş][uü][sş]\b|\bart[iı][sş]\b|\brekor\b",
    re.IGNORECASE)

_PERCENT = re.compile(r"\d\s*%|\d\s*percent|\d\s*por cento|\d\s*pts?\b|\d\s*pontos|\d\s*points?\b",
                      re.IGNORECASE)

# Words that put a headline in the shop, not on the exchange: the price of a
# cup, the supermarket shelf, the consumer price index. The reading is told
# that a shop price is not a market price, so these are read after the rest.
_RETAIL = re.compile(
    r"\bsupermarkets?\b|\bgrocer\w*|\bretail\w*|\bconsumers?\b|\bshops?\b|\bstores?\b|"
    r"\bcups?\b|\binflation\b|\bcpi\b|\bcaf[eé]s\b|\bcoffee ?shops?\b|\bmenu\b|"
    r"\bsupermercados?\b|\bconsumidor(?:es)?\b|\bvarejo\b|\binfla[cç][aã]o\b|\bx[ií]cara\b|"
    r"\btaza\b|\btasse\b|\bчашк\w*",
    re.IGNORECASE)

# At or above this score a headline is treated as a market report and read
# before anything else of its age or older.
REPORT_SCORE = 3
# Places in the reading kept for documents from the day itself that are not
# market reports: the frost, the hurricane, the crop estimate are often
# reported without a word about prices, and on a day with no market report
# retrieved they are the evidence.
KEPT_FOR_THE_DAY = 3


def market_report_score(doc: dict) -> int:
    """0 to 7: how much a headline looks like a report on coffee's market
    price. One point each for naming coffee, a price, and a figure in percent
    or points; two for a word that places it on the exchange (futures,
    settle, arabica, New York, the softs report); two for a market report's
    verb of movement, or one for a weaker word of movement or level. A
    headline about shop prices with nothing from the exchange is held below
    REPORT_SCORE. The score says nothing about which way prices moved."""
    title = doc.get("title") or ""
    move = 2 if _MOVE_REPORT.search(title) else int(bool(_MOVE.search(title)))
    score = (bool(COFFEE.search(title)) + bool(_PRICE.search(title)) + move
             + bool(_PERCENT.search(title)) + 2 * bool(_EXCHANGE.search(title)))
    # A shop-price headline with no word from the exchange is not a market report.
    if _RETAIL.search(title) and not _EXCHANGE.search(title):
        score = min(score, REPORT_SCORE - 1)
    return score


def reading_order(documents: list, age_of, relevance_of, limit: int = None) -> list:
    """The order in which documents are offered to the reading step, which
    reads the first `limit` of them.

    Market reports come first (REPORT_SCORE or more), newest session first,
    the more report-like first within a session. Age is the first key among
    them because the rule decides from the newest session that has evidence,
    so yesterday's reports must not push out today's. Then everything else,
    in the same order - so that a city guide published today does not push
    out yesterday's market report either.

    With a limit, up to KEPT_FOR_THE_DAY of the first `limit` places go to
    the most relevant documents of the newest session that are not market
    reports, when market reports alone would fill them all.

    `age_of` and `relevance_of` are functions of a document; ties are broken
    by document id, never by arrival order."""
    def key(doc):
        return (age_of(doc), -market_report_score(doc), -relevance_of(doc),
                str(doc.get("document_id")))

    reports = sorted((d for d in documents if market_report_score(d) >= REPORT_SCORE), key=key)
    others = sorted((d for d in documents if market_report_score(d) < REPORT_SCORE), key=key)
    if limit is None or len(reports) + min(len(others), KEPT_FOR_THE_DAY) <= limit:
        return reports + others
    newest = min((age_of(d) for d in documents), default=0)
    kept = sorted((d for d in others if age_of(d) == newest),
                  key=lambda d: (-relevance_of(d), str(d.get("document_id"))))[:KEPT_FOR_THE_DAY]
    head = reports[:limit - len(kept)] + kept
    in_head = {id(d) for d in head}
    return head + [d for d in reports + others if id(d) not in in_head]


# The reading is shown a few hundred characters of each document. For nearly
# every document the opening is the right few hundred. For a round-up of
# several commodities it is not - the opening is about sugar or cocoa and the
# paragraph on coffee comes later - and focus_text brings the sentences about
# coffee up behind a shorter opening.
OPENING_CHARS = 600      # what the reading is shown of a document, roughly
LEAD_CHARS = 200         # how much of the opening is kept when passages are brought up
SEPARATOR = " ... "
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")

# Another commodity, or the word for several: the sign of a round-up.
_OTHER_COMMODITY = re.compile(
    r"\bsugar\b|\bcocoa\b|\bcotton\b|\bcrude\b|\boil\b|\bcorn\b|\bwheat\b|\bsoy\w*|"
    r"\bcommodities\b|\bsofts\b|\ba[cç][uú]car\b|\baz[uú]car\b|\bcacau\b|\bcacao\b|"
    r"\balgod[aã]o\b|\balgod[oó]n\b|\bpetr[oó]leo\b|\bsoja\b|\bmilho\b|\btrigo\b|\bsucre\b|"
    r"\bzucker\b|\bkakao\b|[đd][uư][oờ]ng|h[oồ] ti[eê]u|cao su|ca cao|сахар\w*|какао|可可|原糖|砂糖",
    re.IGNORECASE)


def _sentences(text: str) -> list:
    """(start, end) of each sentence, by punctuation; good enough to cut on."""
    bounds, position = [], 0
    for match in _SENTENCE_END.finditer(text):
        bounds.append((position, match.start()))
        position = match.end()
    bounds.append((position, len(text)))
    return [(s, e) for s, e in bounds if e > s]


def _passage_score(sentence: str) -> int:
    """0 if the sentence does not mention coffee; else 1, plus 1 for a word of
    movement and 1 for a figure - the marks of the sentence that says what
    coffee did."""
    if not COFFEE.search(sentence):
        return 0
    return (1 + bool(_MOVE_REPORT.search(sentence) or _MOVE.search(sentence))
            + bool(_PERCENT.search(sentence)))


def focus_text(text: str, limit: int = 4000) -> str:
    """The text as the reading should be shown it.

    Unchanged unless the document is a round-up: its opening names another
    commodity, and a sentence saying what coffee did (coffee with a word of
    movement or a figure) comes after the opening. Then the result is the
    first LEAD_CHARS of the text followed by the sentences about coffee, the
    most telling first, each with the sentence after it (where the reason
    usually is), with " ... " wherever something was left out. Every piece is
    copied exactly, so a quote taken from the result is a quote from the
    document, and the quote check is run against the result.

    An article about coffee is never rearranged: its second sentence may say
    "the contract fell 4.9% as rain was forecast" without the word coffee,
    and cutting to the sentences that have the word would lose it."""
    text = text or ""
    if len(text) <= OPENING_CHARS or not _OTHER_COMMODITY.search(text[:OPENING_CHARS]):
        return text
    sentences = _sentences(text)
    scores = [_passage_score(text[s:e]) for s, e in sentences]
    if not any(score >= 2 and sentences[i][0] >= OPENING_CHARS for i, score in enumerate(scores)):
        return text                     # nothing more about coffee further down
    # The opening kept runs to the end of the sentence LEAD_CHARS falls in
    # (within reason): a first sentence often ends with the reason.
    lead_end = next((e for s, e in sentences if s <= LEAD_CHARS < e), LEAD_CHARS)
    lead_end = min(lead_end, 2 * LEAD_CHARS)
    pieces, used, total = [], set(), 0
    for i in sorted((i for i, score in enumerate(scores) if score), key=lambda i: (-scores[i], i)):
        for j in (i, i + 1):
            if j < len(sentences) and j not in used and sentences[j][0] >= lead_end:
                used.add(j)
                pieces.append(j)
                total += sentences[j][1] - sentences[j][0]
        if total >= limit:
            break
    if not pieces:
        return text                     # all it says about coffee is in the opening
    # Sentences that follow one another in the document are copied as one
    # stretch, whitespace and all, so each piece is an exact substring.
    runs = []
    for j in pieces:
        if runs and j == runs[-1][1] + 1:
            runs[-1][1] = j
        else:
            runs.append([j, j])
    return SEPARATOR.join([text[:lead_end]]
                          + [text[sentences[a][0]:sentences[b][1]] for a, b in runs])


# The exchange's coffee contract opens at 04:15 New York time, which is 08:15
# or 09:15 UTC. A document first seen before 08:00 UTC cannot be reporting
# that day's trading: the Asian morning's "coffee prices today" and the
# overnight wire report of a settlement are both about the session before.
SESSION_OPENS_UTC_HOUR = 8


def session_date(seendate: str) -> Optional[str]:
    """The trading day (YYYY-MM-DD) a document seen at this GDELT-style time
    (YYYYMMDDTHHMMSSZ) can be reporting on, or None if the time is missing.
    Weekends are left to the trading-day arithmetic that uses this date."""
    try:
        seen = datetime.strptime(seendate or "", "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    if seen.hour < SESSION_OPENS_UTC_HOUR:
        seen -= timedelta(days=1)
    return seen.strftime("%Y-%m-%d")
