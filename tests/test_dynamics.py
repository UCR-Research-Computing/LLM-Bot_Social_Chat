"""Tests for the deterministic conversation machinery (no model, no I/O)."""

import pytest

from bot_social_network.dynamics import (
    BM25,
    FairScheduler,
    ObligationLedger,
    RepetitionGuard,
    conversation_metrics,
    find_mentions,
    gini,
    is_duplicate_memory,
    jaccard,
    pick_next,
    questions_for,
    sanitize_post,
    select_memories,
    shingles,
)

NAMES = ["Captain Eva Rostova", "Commander Jax", "Zeke", "Dan", "Steve", "Mike"]


# ---- mentions and questions --------------------------------------------------


def test_find_mentions_multiword_and_order():
    t = "@Zeke check that. @Captain Eva Rostova, and @zeke again"
    assert find_mentions(t, NAMES) == ["Zeke", "Captain Eva Rostova"]


def test_find_mentions_needs_boundary():
    assert find_mentions("@Danny hi", NAMES) == []
    assert find_mentions("@Dan's point", NAMES) == ["Dan"]


def test_questions_for_single_target_takes_every_question():
    t = "@Dan the racks. Can they take 8:1 density? And what about cooling?"
    assert questions_for(t, "Dan", NAMES) == [
        "Can they take 8:1 density?",
        "And what about cooling?",
    ]


def test_questions_for_splits_between_targets():
    t = "@Dan what is the power budget? @Steve which vendor do you trust?"
    assert questions_for(t, "Dan", NAMES) == ["@Dan what is the power budget?"]
    assert questions_for(t, "Steve", NAMES) == ["@Steve which vendor do you trust?"]
    assert questions_for(t, "Mike", NAMES) == []


def test_statement_is_not_a_question():
    assert questions_for("@Dan I agree with you.", "Dan", NAMES) == []


# ---- obligations ---------------------------------------------------------------


def test_ledger_records_and_resolves_by_mentioning_asker():
    led = ObligationLedger()
    led.record(1, "Steve", "@Dan how many GPUs per node?", NAMES, turn=1)
    assert [o.target for o in led.open] == ["Dan"]
    done = led.resolve("Dan", "@Steve eight per node, liquid cooled.", NAMES, turn=2)
    assert len(done) == 1 and not led.open


def test_ledger_resolves_by_topic_overlap_with_single_asker():
    led = ObligationLedger()
    led.record(1, "Steve", "@Dan how many GPUs per node?", NAMES, turn=1)
    led.resolve("Dan", "Eight GPUs fit per node if we go liquid.", NAMES, turn=2)
    assert not led.open


def test_ledger_keeps_question_the_reply_ignored():
    led = ObligationLedger()
    led.record(1, "Steve", "@Dan how many GPUs per node?", NAMES, turn=1)
    led.record(2, "Mike", "@Dan what storage vendor?", NAMES, turn=2)
    led.resolve("Dan", "@Mike Ceph on NVMe, no vendor lock-in.", NAMES, turn=3)
    assert [(o.asker, o.target) for o in led.open] == [("Steve", "Dan")]


def test_ledger_ignores_self_mention_and_expires():
    led = ObligationLedger(max_age=3)
    led.record(1, "Dan", "@Dan am I talking to myself?", NAMES, turn=1)
    assert not led.open
    led.record(2, "Steve", "@Mike are you there?", NAMES, turn=2)
    assert led.expire(turn=6) and not led.open


def test_debtors_oldest_first():
    led = ObligationLedger()
    led.record(1, "Steve", "@Mike storage?", NAMES, turn=1)
    led.record(2, "Steve", "@Dan power?", NAMES, turn=2)
    led.record(3, "Zeke", "@Mike again, storage?", NAMES, turn=3)
    assert led.debtors() == [("Mike", 1), ("Dan", 2)]


# ---- scheduling -----------------------------------------------------------------


def test_fair_scheduler_favors_bot_that_said_less():
    f = FairScheduler()
    f.sync(["Dan", "Steve", "Mike"])
    f.spend("Dan", 120)
    f.spend("Steve", 20)
    assert f.ranked(["Dan", "Steve", "Mike"]) == ["Mike", "Steve", "Dan"]


def test_fair_scheduler_ties_go_to_longest_silent():
    f = FairScheduler()
    f.sync(["A", "B", "C"])
    f.spend("A", 10)
    f.spend("B", 10)
    f.spend("C", 10)
    assert f.ranked(["A", "B", "C"]) == ["A", "B", "C"]


def test_fair_scheduler_late_joiner_starts_level():
    f = FairScheduler()
    f.sync(["A", "B"])
    f.spend("A", 300)
    f.spend("B", 200)
    f.sync(["A", "B", "New"])
    assert f.words["New"] == 200  # level with the quietest, not 0
    f.sync(["A"])
    assert "B" not in f.last_turn and "New" not in f.words


def test_fair_scheduler_evens_out_over_time():
    import random

    rng = random.Random(0)
    f = FairScheduler()
    names = ["A", "B", "C", "D"]
    f.sync(names)
    last = None
    for _ in range(400):
        pool = [n for n in names if n != last]
        who = f.ranked(pool)[0]
        f.spend(who, rng.randint(20, 90))
        last = who
    spread = max(f.words.values()) - min(f.words.values())
    assert spread <= 90  # never more than one long post apart


def test_pick_next_priority_order():
    led = ObligationLedger()
    fair = FairScheduler()
    fair.sync(["Dan", "Steve", "Mike"])
    fair.spend("Mike", 500)
    fair.spend("Dan", 100)
    # no debts: most credit wins, never the last speaker
    assert pick_next(["Dan", "Steve", "Mike"], led, fair, last_speaker="Dan") == "Steve"
    # a debt beats fairness
    led.record(1, "Dan", "@Mike storage plan?", NAMES, turn=1)
    assert pick_next(["Dan", "Steve", "Mike"], led, fair, last_speaker="Dan") == "Mike"
    # but not if the debtor just spoke
    assert pick_next(["Dan", "Steve", "Mike"], led, fair, last_speaker="Mike") != "Mike"


# ---- repetition ------------------------------------------------------------------


def test_shingles_and_jaccard():
    a = shingles("the cooling loop is failing again")
    b = shingles("the cooling loop is failing now")
    assert 0.5 < jaccard(a, b) < 1
    assert jaccard(a, shingles("totally unrelated words here")) == 0


def test_repetition_guard_flags_near_duplicate_only():
    g = RepetitionGuard()
    g.add("Dan", "We should put the GPUs in rack four because it has the most cooling.")
    assert g.is_repeat(
        "We should put the GPUs in rack four because it has more cooling."
    )
    assert not g.is_repeat("Storage is the real bottleneck, not compute.")
    score, who = g.check("We should put the GPUs in rack four because it has cooling")
    assert who == "Dan" and score > 0.35


def test_repetition_window_forgets_old_posts():
    g = RepetitionGuard(window=2)
    g.add("A", "alpha beta gamma delta epsilon")
    g.add("B", "one two three four five")
    g.add("C", "six seven eight nine ten")
    assert not g.is_repeat("alpha beta gamma delta epsilon")


# ---- retrieval --------------------------------------------------------------------


def test_bm25_ranks_relevant_doc_first():
    docs = [
        "Steve prefers TPUs for training",
        "Mike worries about storage throughput",
        "The dean approved the budget",
    ]
    s = BM25(docs).score("what about storage throughput for the cluster")
    assert max(range(3), key=lambda i: s[i]) == 1


def test_select_memories_keeps_seeds_then_relevant_then_newest():
    mems = [("persona", "I am Dan, a sysadmin"), ("style", "blunt")]
    mems += [(f"note {i}", f"filler fact number {i}") for i in range(20)]
    mems.append(("cooling", "rack four has liquid cooling for GPUs"))
    idx = select_memories(mems, "can rack four take the GPUs", k=5)
    assert idx[:2] == [0, 1]
    assert len(mems) - 1 in idx  # the relevant one
    assert len(idx) == 5 and idx == sorted(idx)


def test_select_memories_returns_all_when_few():
    assert select_memories([("a", "b"), ("c", "d")], "x", k=8) == [0, 1]


def test_duplicate_memory_detection():
    have = [("Steve on TPUs", "Steve wants TPUs for training")]
    assert is_duplicate_memory(
        ("Steve TPUs", "Steve wants TPUs for training jobs"), have
    )
    assert not is_duplicate_memory(("Mike", "Mike distrusts cloud credits"), have)
    assert is_duplicate_memory(("", "the and of"), have)  # no content words


# ---- sanitizing -------------------------------------------------------------------


def test_sanitize_drops_fabricated_lines_and_self_prefix():
    raw = "Dan: I think liquid cooling wins.\nSteve: No way, air is fine.\nDan: ok"
    s = sanitize_post(raw, "Dan", NAMES)
    assert s.text == "I think liquid cooling wins."
    assert any("Steve" in f for f in s.fixes)


def test_sanitize_fixes_misspelled_and_unknown_mentions():
    s = sanitize_post("@Stve and @Zek, ask @Bob too.", "Dan", NAMES)
    assert s.text == "@Steve and @Zeke, ask Bob too."


def test_sanitize_keeps_valid_multiword_mention():
    s = sanitize_post("@Captain Eva Rostova, drive is ready.", "Zeke", NAMES)
    assert s.text == "@Captain Eva Rostova, drive is ready." and not s.fixes


def test_sanitize_trims_at_sentence_boundary():
    raw = "One two three. " * 20 + "and then it trails off without end"
    s = sanitize_post(raw, "Dan", NAMES, max_words=20)
    assert s.text.endswith(".") and len(s.text.split()) <= 20


@pytest.mark.parametrize("raw", ['"Quoted post."', "'Quoted post.'"])
def test_sanitize_strips_quotes(raw):
    assert sanitize_post(raw, "Dan", NAMES).text == "Quoted post."


# ---- metrics ----------------------------------------------------------------------


def test_gini():
    assert gini([10, 10, 10]) == 0
    assert gini([0, 0, 30]) == pytest.approx(2 / 3)
    assert gini([]) == 0


def test_conversation_metrics_on_a_known_transcript():
    posts = [
        ("SYSTEM", "Topic: new cluster"),
        ("Steve", "@Dan can the racks take GPUs?"),
        ("Dan", "@Steve yes, rack four has liquid cooling for GPUs."),
        ("Mike", "@Steve what storage do we pair with it?"),
        ("Dan", "Storage should be Ceph."),
        ("Mike", "@Zeke are you there?"),
    ]
    m = conversation_metrics(posts, ["Steve", "Dan", "Mike", "Zeke"])
    assert m["posts"] == 5
    assert m["mentions"] == 4
    # Steve->Dan got a reply; Dan->Steve, Mike->Steve, Mike->Zeke did not
    assert m["mention_reply_rate"] == 0.25
    assert m["questions"] == 3
    assert m["questions_answered"] == 1
    assert m["open_questions"] == 2
    # pairs: Steve->Dan, Dan->Steve, Mike->Steve, Mike->Zeke; only Steve<->Dan is mutual
    assert m["reciprocity"] == 0.5
    assert 0 < m["speaking_gini"] < 1
    assert m["repeat_rate"] == 0
