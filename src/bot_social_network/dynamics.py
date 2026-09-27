"""Deterministic conversation machinery: no model calls, no I/O, stdlib only.

Everything here is plain computer science wrapped around the model so it has
less to guess:

- mention/question detection        (text scanning against the member list)
- ObligationLedger                  (priority queue of open @mention questions)
- FairScheduler                     (least-attained-service fair queuing)
- shingles / jaccard / Repetition   (w-shingling near-duplicate detection)
- BM25                              (Okapi BM25 ranking for memory retrieval)
- sanitize_post                     (rule-based cleanup of model output)
- conversation_metrics              (reply rate, latency, reciprocity, Gini,
                                     repetition, lexical diversity)

The module has a small, pure interface so it could be swapped for a compiled
implementation later without touching callers.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- text

_WORD = re.compile(r"[a-z0-9']+")
STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for with from as is are was
    were be been being it its this that these those i you he she we they me him her
    us them my your his our their what which who whom whose do does did have has had
    not no yes can could would should will just about into over than too very also
    there here how why when where all any some more most such only own same up out
    off again further once am im ive youre dont cant wont lets let get got""".split()
)


def tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def content_words(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]


def find_mentions(text: str, names: Sequence[str]) -> list[str]:
    """Member names @mentioned in text, in order of first appearance.

    Matches the known names longest-first, so "@Captain Eva Rostova" is one
    mention (a regex like @(\\w+) would stop at "Captain"). Case-insensitive;
    requires a word boundary after the name so "@Danny" is not "@Dan".
    """
    low = (text or "").lower()
    found: list[tuple[int, str]] = []
    taken: set[int] = set()
    for n in sorted(names, key=len, reverse=True):
        key = "@" + n.lower()
        start = 0
        while (i := low.find(key, start)) >= 0:
            end = i + len(key)
            boundary = end == len(low) or not (low[end].isalnum() or low[end] == "_")
            if boundary and i not in taken:
                found.append((i, n))
                taken.update(range(i, end))
            start = end
    return list(dict.fromkeys(n for _, n in sorted(found)))


_SENT = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def questions_for(text: str, target: str, names: Sequence[str]) -> list[str]:
    """Question sentences in `text` aimed at `target`.

    A question is aimed at a bot if the sentence mentions it, or the post
    mentions only that bot. Posts that mention several bots only assign a
    question to the bots named in that sentence (or the nearest preceding
    mention).
    """
    mentioned = find_mentions(text, names)
    if target not in mentioned:
        return []
    out: list[str] = []
    current: str | None = mentioned[0] if len(mentioned) == 1 else None
    for s in sentences(text):
        in_s = find_mentions(s, names)
        if in_s:
            current = in_s[-1]
        if s.endswith("?") and (target in in_s or (not in_s and current == target)):
            out.append(s)
    return out


# --------------------------------------------------------------------------- obligations


@dataclass
class Obligation:
    post_id: int
    asker: str
    target: str
    question: str
    turn: int  # turn at which it was asked
    keywords: frozenset[str]


@dataclass
class ObligationLedger:
    """Open questions: who owes whom an answer. A priority queue by age."""

    open: list[Obligation] = field(default_factory=list)
    answered: list[tuple[Obligation, int]] = field(default_factory=list)  # (ob, turn)
    expired: list[Obligation] = field(default_factory=list)
    max_age: int = 12  # turns before an unanswered question is dropped

    def record(
        self, post_id: int, sender: str, text: str, names: Sequence[str], turn: int
    ) -> list[Obligation]:
        new = []
        for target in find_mentions(text, names):
            if target == sender:
                continue
            for q in questions_for(text, target, names):
                ob = Obligation(
                    post_id, sender, target, q, turn, frozenset(content_words(q))
                )
                self.open.append(ob)
                new.append(ob)
        return new

    def resolve(
        self, speaker: str, text: str, names: Sequence[str], turn: int
    ) -> list[Obligation]:
        """Mark obligations of `speaker` as answered by this post.

        An open question to the speaker counts as answered when the reply
        @mentions the asker, or (single-asker case) when it shares content
        words with the question. Posting at all clears obligations to the
        asker the speaker addressed; others stay open.
        """
        mine = [o for o in self.open if o.target == speaker]
        if not mine:
            return []
        addressed = set(find_mentions(text, names))
        words = set(content_words(text))
        askers = {o.asker for o in mine}
        done = []
        for o in mine:
            overlap = len(o.keywords & words)
            if o.asker in addressed or (len(askers) == 1 and overlap >= 1):
                done.append(o)
        for o in done:
            self.open.remove(o)
            self.answered.append((o, turn))
        return done

    def expire(self, turn: int) -> list[Obligation]:
        old = [o for o in self.open if turn - o.turn > self.max_age]
        for o in old:
            self.open.remove(o)
            self.expired.append(o)
        return old

    def owed_by(self, name: str) -> list[Obligation]:
        return sorted((o for o in self.open if o.target == name), key=lambda o: o.turn)

    def debtors(self) -> list[tuple[str, int]]:
        """(bot, turn of its oldest open question), oldest first."""
        oldest: dict[str, int] = {}
        for o in self.open:
            oldest[o.target] = min(oldest.get(o.target, o.turn), o.turn)
        return sorted(oldest.items(), key=lambda kv: kv[1])

    def __len__(self) -> int:
        return len(self.open)


# --------------------------------------------------------------------------- scheduling


@dataclass
class FairScheduler:
    """Least-attained-service fair queuing by words spoken.

    The fairest next speaker is the one that has said the fewest words so far;
    ties go to whoever has been silent longest. A bot that writes long posts
    therefore waits longer before speaking again, and a quiet bot rises to the
    front. A bot that joins late starts level with the quietest member instead
    of at zero, so it does not monopolize the floor to catch up.
    """

    words: Counter[str] = field(default_factory=Counter)
    last_turn: dict[str, int] = field(default_factory=dict)
    turn: int = 0

    def sync(self, names: Iterable[str]) -> None:
        names = list(names)
        floor = min((self.words[n] for n in names if n in self.last_turn), default=0)
        for n in names:
            if n not in self.last_turn:
                self.words[n] = floor
                self.last_turn[n] = -1
        for n in list(self.last_turn):
            if n not in names:
                del self.last_turn[n]
                self.words.pop(n, None)

    def spend(self, name: str, n_words: int) -> None:
        self.turn += 1
        self.words[name] += n_words
        self.last_turn[name] = self.turn

    def ranked(self, candidates: Sequence[str]) -> list[str]:
        return sorted(
            candidates,
            key=lambda n: (self.words.get(n, 0), self.last_turn.get(n, -1)),
        )


def pick_next(
    candidates: Sequence[str],
    ledger: ObligationLedger,
    fair: FairScheduler,
    last_speaker: str | None,
) -> str | None:
    """Deterministic choice: the bot with the oldest unanswered question goes
    first; otherwise the bot with the most fair-share credit. Never the same
    bot twice in a row unless it is the only one."""
    pool = [c for c in candidates if c != last_speaker] or list(candidates)
    if not pool:
        return None
    for name, _turn in ledger.debtors():
        if name in pool:
            return name
    return fair.ranked(pool)[0]


# --------------------------------------------------------------------------- repetition


def shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    """Word k-shingles (w-shingling, Broder 1997)."""
    t = tokens(text)
    if len(t) < k:
        return {tuple(t)} if t else set()
    return {tuple(t[i : i + k]) for i in range(len(t) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class RepetitionGuard:
    """Flags a post that is a near-duplicate of a recent one (or one of the
    same bot's own recent posts) using Jaccard similarity of 3-word shingles."""

    window: int = 30
    threshold: float = 0.35
    history: deque[tuple[str, set[tuple[str, ...]]]] = field(
        default_factory=lambda: deque(maxlen=30)
    )

    def __post_init__(self) -> None:
        self.history = deque(self.history, maxlen=self.window)

    def check(self, text: str) -> tuple[float, str | None]:
        """(highest similarity, sender of the most similar post)."""
        sh = shingles(text)
        best, who = 0.0, None
        for sender, prev in self.history:
            j = jaccard(sh, prev)
            if j > best:
                best, who = j, sender
        return best, who

    def is_repeat(self, text: str) -> bool:
        return self.check(text)[0] >= self.threshold

    def add(self, sender: str, text: str) -> None:
        self.history.append((sender, shingles(text)))


# --------------------------------------------------------------------------- retrieval


class BM25:
    """Okapi BM25 over short documents (bot memories)."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [content_words(d) for d in docs]
        self.k1, self.b = k1, b
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query: str) -> list[float]:
        q = content_words(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            norm = self.k1 * (
                1 - self.b + self.b * (len(d) / self.avgdl if self.avgdl else 0)
            )
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


def select_memories(
    memories: Sequence[tuple[str, str]],
    query: str,
    k: int = 8,
    keep_first: int = 2,
) -> list[int]:
    """Indices of the memories to show: the first `keep_first` (seeded core
    identity), then the top BM25 matches for the recent conversation, then the
    newest ones to fill up to k. Returned in original order."""
    n = len(memories)
    if n <= k:
        return list(range(n))
    chosen: list[int] = list(range(min(keep_first, n)))
    scores = BM25([f"{key} {val}" for key, val in memories]).score(query)
    for i in sorted(range(n), key=lambda i: -scores[i]):
        if len(chosen) >= k:
            break
        if scores[i] > 0 and i not in chosen:
            chosen.append(i)
    for i in range(n - 1, -1, -1):
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def is_duplicate_memory(
    new: tuple[str, str], existing: Sequence[tuple[str, str]], threshold: float = 0.6
) -> bool:
    """True if the new memory's content words overlap an existing one heavily."""
    a = set(content_words(" ".join(new)))
    if not a:
        return True
    return any(
        jaccard(a, set(content_words(" ".join(e)))) >= threshold for e in existing
    )


# --------------------------------------------------------------------------- sanitizing


@dataclass
class Sanitized:
    text: str
    fixes: list[str]


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance with an early exit once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def sanitize_post(
    text: str, speaker: str, names: Sequence[str], max_words: int = 110
) -> Sanitized:
    """Rule-based cleanup of a model reply.

    - drops any line the model wrote for another member ("Mike: ...")
    - strips a leading self-prefix ("Dan:", "**Dan:**", "@Dan:")
    - fixes @mentions that are one or two edits from a member name, and
      unwraps @mentions of people who are not in the chat
    - collapses whitespace, removes surrounding quotes, trims to max_words at
      a sentence boundary
    """
    fixes: list[str] = []
    t = (text or "").strip()
    others = [n for n in names if n != speaker]

    lines = t.splitlines()
    kept = []
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("*").lstrip("@")
        hit = next(
            (n for n in others if stripped.lower().startswith(n.lower() + ":")), None
        )
        if hit and i > 0:
            fixes.append(f"removed line written as {hit}")
            break  # everything after is a fabricated continuation
        kept.append(line)
    t = "\n".join(kept).strip()

    for prefix in (f"**{speaker}:**", f"@{speaker}:", f"{speaker}:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix) :].strip()
            fixes.append("removed self prefix")
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
        fixes.append("removed quotes")

    def fix_mention(m: re.Match[str]) -> str:
        raw = m.group(1)
        if find_mentions("@" + raw, names):
            return m.group(0)
        word = raw.rstrip(".,!?;:")
        tail = raw[len(word) :]
        best = min(
            names, key=lambda n: _edit_distance(word.lower(), n.lower()), default=None
        )
        if best and _edit_distance(word.lower(), best.lower()) <= (
            1 if len(best) <= 4 else 2
        ):
            fixes.append(f"@{word} -> @{best}")
            return f"@{best}{tail}"
        # a multi-word member name whose first word matches is handled by
        # find_mentions above; anything else is not a member
        if not any(n.lower().startswith(word.lower()) for n in names):
            fixes.append(f"unknown mention @{word}")
            return word + tail
        return m.group(0)

    t = re.sub(r"@([A-Za-z0-9_][A-Za-z0-9_.'-]*)", fix_mention, t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    words = t.split()
    if len(words) > max_words:
        cut = " ".join(words[:max_words])
        sents = sentences(cut)
        if len(sents) > 1 and not cut.rstrip().endswith((".", "!", "?")):
            cut = " ".join(sents[:-1])
        t = cut.strip()
        fixes.append(f"trimmed to {len(t.split())} words")
    return Sanitized(t, fixes)


# --------------------------------------------------------------------------- metrics


def gini(values: Sequence[float]) -> float:
    """Gini coefficient: 0 = perfectly even, ->1 = one speaker dominates."""
    v = sorted(x for x in values if x >= 0)
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cum) / (n * sum(v)) - (n + 1) / n


def conversation_metrics(
    posts: Sequence[tuple[str, str]], names: Sequence[str]
) -> dict[str, float | int]:
    """Measure a transcript without any model.

    posts: (sender, text) oldest first; SYSTEM posts are context only.
    """
    bots = [(i, s, t) for i, (s, t) in enumerate(posts) if s != "SYSTEM"]
    ledger = ObligationLedger(max_age=10**9)
    mentions = replied = 0
    latencies: list[int] = []
    pairs: Counter[tuple[str, str]] = Counter()
    guard = RepetitionGuard(threshold=0.35)
    repeats = 0
    all_tokens: list[str] = []
    words_by: Counter[str] = Counter({n: 0 for n in names})
    for turn, (i, s, t) in enumerate(bots):
        ledger.resolve(s, t, names, turn)
        ledger.record(i, s, t, names, turn)
        m = [x for x in find_mentions(t, names) if x != s]
        for target in m:
            mentions += 1
            pairs[(s, target)] += 1
            nxt = next(
                (k for k in range(turn + 1, len(bots)) if bots[k][1] == target), None
            )
            if nxt is not None:
                replied += 1
                latencies.append(nxt - turn)
        if guard.is_repeat(t):
            repeats += 1
        guard.add(s, t)
        toks = tokens(t)
        all_tokens += toks
        words_by[s] += len(toks)
    q_total = len(ledger.answered) + len(ledger.open)
    recip_pairs = [p for p in pairs if (p[1], p[0]) in pairs]
    uniq = len(set(all_tokens))
    return {
        "posts": len(bots),
        "mentions": mentions,
        "mention_reply_rate": round(replied / mentions, 3) if mentions else 0.0,
        "mean_reply_latency": round(sum(latencies) / len(latencies), 2)
        if latencies
        else 0.0,
        "questions": q_total,
        "questions_answered": len(ledger.answered),
        "question_answer_rate": round(len(ledger.answered) / q_total, 3)
        if q_total
        else 0.0,
        "open_questions": len(ledger.open),
        "reciprocity": round(len(recip_pairs) / len(pairs), 3) if pairs else 0.0,
        "speaking_gini": round(gini(list(words_by.values())), 3),
        "repeat_rate": round(repeats / len(bots), 3) if bots else 0.0,
        "type_token_ratio": round(uniq / len(all_tokens), 3) if all_tokens else 0.0,
    }
