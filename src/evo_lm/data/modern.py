"""Build a licensed, modern-Chinese training mix with compositional tasks.

The generated train/validation files keep all variants derived from the same
source record on the same side of the split.  This prevents a natural answer
from appearing in validation while a word-combination or reorder variant of
that answer is present in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ONLY_CJK_RE = re.compile(r"^[\u3400-\u9fff]+$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n+")
_SPACE_RE = re.compile(r"[ \t\u3000]+")
_BLANK_RE = re.compile(r"\n\s*\n")

# Small models readily memorize boilerplate and low-quality forum artifacts.
_REJECT_FRAGMENTS = (
    "http://",
    "https://",
    "www.",
    "点击链接",
    "请给好评",
    "求好评",
    "百度一下",
    "楼主",
    "楼上",
    "复制粘贴",
    "<script",
    "javascript:",
)

_POS_ZH = {
    "a": "形容词",
    "ad": "副词性形容词",
    "an": "名词性形容词",
    "c": "连词",
    "d": "副词",
    "e": "感叹词",
    "f": "方位词",
    "i": "成语",
    "l": "固定表达",
    "m": "数词",
    "n": "名词",
    "nr": "人名",
    "ns": "地名",
    "o": "拟声词",
    "p": "介词",
    "q": "量词",
    "r": "代词",
    "s": "处所词",
    "t": "时间词",
    "u": "助词",
    "v": "动词",
    "vd": "副词性动词",
    "vn": "名词性动词",
    "y": "语气词",
    "z": "状态词",
}


@dataclass
class SourceGroup:
    """Blocks tied to one source record for leakage-free splitting."""

    key: str
    source: str
    blocks: list[str] = field(default_factory=list)


@dataclass
class BuildStats:
    groups: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    blocks: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rejected: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def normalize_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    # Blank lines are the corpus document delimiter, so external paragraphs
    # must use a single newline inside one source group.
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chinese_fraction(text: str) -> float:
    meaningful = sum(ch.isalnum() or _CJK_RE.match(ch) is not None for ch in text)
    if meaningful == 0:
        return 0.0
    return len(_CJK_RE.findall(text)) / meaningful


def _usable(text: str, *, min_chars: int, max_chars: int, min_zh: float) -> bool:
    lowered = text.lower()
    return (
        min_chars <= len(text) <= max_chars
        and chinese_fraction(text) >= min_zh
        and "�" not in text
        and not any(fragment in lowered for fragment in _REJECT_FRAGMENTS)
    )


def _dialogue(question: str, answer: str) -> str:
    return f"用户: {question}\n助手: {answer}"


def _stable_bucket(seed: int, key: str, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_lines(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _reservoir(items: Iterable[SourceGroup], limit: int, seed: int) -> list[SourceGroup]:
    rng = random.Random(seed)
    sample: list[SourceGroup] = []
    for seen, item in enumerate(items, start=1):
        if len(sample) < limit:
            sample.append(item)
            continue
        position = rng.randrange(seen)
        if position < limit:
            sample[position] = item
    return sample


def load_hsk(path: Path, *, max_words: int = 8_000) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    words: list[dict] = []
    for item in raw:
        word = normalize_text(item.get("simplified"))
        forms = item.get("forms") or []
        transcriptions = forms[0].get("transcriptions", {}) if forms else {}
        pinyin = normalize_text(transcriptions.get("pinyin"))
        pos = [code for code in item.get("pos") or [] if code in _POS_ZH]
        if not (1 <= len(word) <= 4 and _ONLY_CJK_RE.fullmatch(word) and pinyin):
            continue
        words.append(
            {
                "word": word,
                "pinyin": pinyin,
                "pos": pos,
                "frequency": int(item.get("frequency") or 10**9),
            }
        )
    words.sort(key=lambda row: (row["frequency"], len(row["word"]), row["word"]))
    return words[:max_words]


def hsk_groups(entries: list[dict], *, limit: int) -> Iterator[SourceGroup]:
    emitted = 0
    for entry in entries:
        if not entry["pos"]:
            continue
        labels = list(dict.fromkeys(_POS_ZH[code] for code in entry["pos"]))[:3]
        label = "、".join(labels)
        word = entry["word"]
        pinyin = entry["pinyin"]
        yield SourceGroup(
            key=f"hsk:{word}",
            source="hsk_dictionary",
            blocks=[
                _dialogue(
                    f"“{word}”怎么读，通常是什么词性？",
                    f"“{word}”读作“{pinyin}”，通常可作{label}。",
                )
            ],
        )
        emitted += 1
        if emitted >= limit:
            return


def hc3_groups(path: Path, stats: BuildStats) -> Iterator[SourceGroup]:
    allowed_sources = {"baike", "open_qa", "nlpcc_dbqa"}
    for index, row in enumerate(_json_lines(path)):
        source = normalize_text(row.get("source"))
        question = normalize_text(row.get("question"))
        answers = [normalize_text(answer) for answer in row.get("human_answers") or []]
        answers = [
            answer
            for answer in answers
            if _usable(answer, min_chars=8, max_chars=480, min_zh=0.45)
        ]
        if source not in allowed_sources or not _usable(
            question, min_chars=4, max_chars=180, min_zh=0.35
        ) or not answers:
            stats.rejected["hc3"] += 1
            continue
        # Prefer a substantial answer without always selecting the longest essay.
        answer = min(answers, key=lambda value: (abs(len(value) - 180), len(value)))
        yield SourceGroup(
            key=f"hc3:{index}:{hashlib.sha256(question.encode()).hexdigest()[:12]}",
            source="hc3_human",
            blocks=[_dialogue(question, answer)],
        )


def coig_groups(path: Path, stats: BuildStats) -> Iterator[SourceGroup]:
    for index, row in enumerate(_json_lines(path)):
        instruction = normalize_text(row.get("trans_instruction"))
        context = normalize_text(row.get("trans_input"))
        answer = normalize_text(row.get("trans_output"))
        if context and context.lower() not in {"无", "没有", "none", "null", "n/a"}:
            question = f"{instruction}\n补充信息：{context}"
        else:
            question = instruction
        if not _usable(question, min_chars=6, max_chars=360, min_zh=0.4) or not _usable(
            answer, min_chars=6, max_chars=520, min_zh=0.45
        ):
            stats.rejected["coig"] += 1
            continue
        if "```" in question or "```" in answer:
            stats.rejected["coig"] += 1
            continue
        yield SourceGroup(
            key=f"coig:{index}:{hashlib.sha256(question.encode()).hexdigest()[:12]}",
            source="coig_instruction",
            blocks=[_dialogue(question, answer)],
        )


def coig_value_groups(path: Path, stats: BuildStats) -> Iterator[SourceGroup]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for index, row in enumerate(raw):
        instruction = normalize_text(row.get("instruction"))
        context = normalize_text(row.get("input"))
        answer = normalize_text(row.get("output"))
        question = f"{instruction}\n补充信息：{context}" if context else instruction
        if not _usable(question, min_chars=4, max_chars=280, min_zh=0.45) or not _usable(
            answer, min_chars=8, max_chars=520, min_zh=0.5
        ):
            stats.rejected["coig_value"] += 1
            continue
        yield SourceGroup(
            key=f"coig-value:{index}:{hashlib.sha256(question.encode()).hexdigest()[:12]}",
            source="coig_value_alignment",
            blocks=[_dialogue(question, answer)],
        )


def legacy_groups(path: Path, stats: BuildStats) -> Iterator[SourceGroup]:
    text = path.read_text(encoding="utf-8")
    for index, raw in enumerate(_BLANK_RE.split(text)):
        block = normalize_text(raw)
        if not _usable(block, min_chars=8, max_chars=1_200, min_zh=0.35):
            stats.rejected["legacy"] += 1
            continue
        yield SourceGroup(
            key=f"legacy:{index}:{hashlib.sha256(block.encode()).hexdigest()[:12]}",
            source="legacy_speak_mix",
            blocks=[block],
        )


def _sentence_candidates(answer: str) -> list[str]:
    candidates = []
    for raw in _SENTENCE_SPLIT_RE.split(answer):
        sentence = raw.strip(" \n\t‘’“”\"'")
        if _usable(sentence, min_chars=12, max_chars=96, min_zh=0.6):
            candidates.append(sentence)
    return candidates


def _lexicon_index(entries: list[dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        word = entry["word"]
        if len(word) >= 2:
            index[word[0]].append(word)
    for words in index.values():
        words.sort(key=lambda value: (-len(value), value))
    return index


def _words_in_sentence(sentence: str, index: dict[str, list[str]]) -> list[str]:
    found = []
    seen = set()
    for char in sentence:
        for word in index.get(char, []):
            if word in sentence and word not in seen:
                found.append(word)
                seen.add(word)
                if len(found) >= 8:
                    return found
    return found


def _shuffled_chunks(sentence: str, rng: random.Random) -> list[str] | None:
    core = sentence.rstrip("。！？!?；;")
    if not 14 <= len(core) <= 72:
        return None
    chunks = []
    cursor = 0
    while cursor < len(core):
        width = rng.randint(2, 4)
        chunks.append(core[cursor : cursor + width])
        cursor += width
    if not 4 <= len(chunks) <= 18:
        return None
    shuffled = chunks[:]
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != chunks:
            return shuffled
    return None


def add_compositional_variants(
    groups: Iterable[SourceGroup],
    lexicon: list[dict],
    *,
    seed: int,
    combination_rate: float = 0.20,
    reorder_rate: float = 0.12,
) -> list[SourceGroup]:
    index = _lexicon_index(lexicon)
    output = []
    for group in groups:
        if not group.blocks or "\n助手: " not in group.blocks[0]:
            output.append(group)
            continue
        answer = group.blocks[0].split("\n助手: ", 1)[1]
        sentences = _sentence_candidates(answer)
        if not sentences:
            output.append(group)
            continue
        rng = random.Random(_stable_bucket(seed, group.key, buckets=2**31 - 1))
        sentence = rng.choice(sentences)
        if rng.random() < combination_rate:
            words = _words_in_sentence(sentence, index)
            if len(words) >= 2:
                chosen = rng.sample(words, 2)
                group.blocks.append(
                    _dialogue(
                        f"请用“{chosen[0]}”和“{chosen[1]}”组成一句自然的话。",
                        sentence,
                    )
                )
        if rng.random() < reorder_rate:
            chunks = _shuffled_chunks(sentence, rng)
            if chunks:
                group.blocks.append(
                    _dialogue(
                        f"请把这些片段排列成自然句子：{'｜'.join(chunks)}",
                        sentence,
                    )
                )
        output.append(group)
    return output


def _dedupe_and_split(
    groups: Iterable[SourceGroup],
    *,
    seed: int,
    val_ratio: float,
    stats: BuildStats,
) -> tuple[list[str], list[str]]:
    seen = set()
    train: list[str] = []
    val: list[str] = []
    threshold = round(val_ratio * 10_000)
    for group in groups:
        destination = val if _stable_bucket(seed, group.key) < threshold else train
        added = 0
        for raw in group.blocks:
            block = normalize_text(raw)
            key = hashlib.sha256(block.encode()).digest()
            if key in seen:
                stats.rejected["duplicate"] += 1
                continue
            seen.add(key)
            destination.append(block)
            added += 1
            stats.blocks[group.source] += 1
        if added:
            stats.groups[group.source] += 1
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_modern_mix(
    *,
    legacy_path: Path,
    hc3_path: Path,
    coig_path: Path,
    coig_value_path: Path | None,
    hsk_path: Path,
    train_out: Path,
    val_out: Path,
    manifest_out: Path,
    seed: int = 3307,
    val_ratio: float = 0.025,
    coig_limit: int = 10_000,
    hsk_limit: int = 4_000,
) -> dict:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")
    stats = BuildStats()
    lexicon = load_hsk(hsk_path)

    legacy = list(legacy_groups(legacy_path, stats))
    hc3 = add_compositional_variants(
        hc3_groups(hc3_path, stats), lexicon, seed=seed + 1
    )
    coig_sample = _reservoir(coig_groups(coig_path, stats), coig_limit, seed + 2)
    coig = add_compositional_variants(coig_sample, lexicon, seed=seed + 3)
    coig_value = (
        add_compositional_variants(
            coig_value_groups(coig_value_path, stats), lexicon, seed=seed + 4
        )
        if coig_value_path is not None
        else []
    )
    dictionary = list(hsk_groups(lexicon, limit=hsk_limit))

    train, val = _dedupe_and_split(
        [*legacy, *hc3, *coig, *coig_value, *dictionary],
        seed=seed,
        val_ratio=val_ratio,
        stats=stats,
    )
    if not train or not val:
        raise ValueError("modern mix produced an empty train or validation split")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text("\n\n".join(train) + "\n", encoding="utf-8")
    val_out.write_text("\n\n".join(val) + "\n", encoding="utf-8")

    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_blocks": len(train),
        "val_blocks": len(val),
        "train_chars": sum(map(len, train)),
        "val_chars": sum(map(len, val)),
        "groups_by_source": dict(stats.groups),
        "blocks_by_source": dict(stats.blocks),
        "rejected": dict(stats.rejected),
        "sources": [
            {
                "name": "existing speak_mix",
                "path": str(legacy_path),
                "license": "project-local source mix; retain upstream notices",
                "sha256": _file_sha256(legacy_path),
            },
            {
                "name": "HC3-Chinese",
                "url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese",
                "license": "CC-BY-SA-4.0",
                "selection": "human_answers from baike/open_qa/nlpcc_dbqa only",
                "sha256": _file_sha256(hc3_path),
            },
            {
                "name": "BAAI COIG translated instructions",
                "url": "https://huggingface.co/datasets/BAAI/COIG",
                "license": "Apache-2.0 (see dataset card for component notices)",
                "selection": f"deterministic reservoir sample, limit={coig_limit}",
                "sha256": _file_sha256(coig_path),
            },
            *(
                [
                    {
                        "name": "BAAI COIG human value alignment part 1",
                        "url": "https://huggingface.co/datasets/BAAI/COIG",
                        "license": "Apache-2.0",
                        "selection": "all records passing Chinese/length quality filters",
                        "sha256": _file_sha256(coig_value_path),
                    }
                ]
                if coig_value_path is not None
                else []
            ),
            {
                "name": "Complete HSK Vocabulary",
                "url": "https://github.com/drkameleon/complete-hsk-vocabulary",
                "license": "MIT; dictionary definitions derive from CC-CEDICT",
                "selection": f"frequency-ranked pronunciation/POS tasks, limit={hsk_limit}",
                "sha256": _file_sha256(hsk_path),
            },
        ],
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build licensed modern Chinese EvoLM mix")
    parser.add_argument("--legacy", default="data/processed/speak_mix.txt")
    parser.add_argument("--hc3", default="data/raw/modern/hc3/all.jsonl")
    parser.add_argument(
        "--coig", default="data/raw/modern/coig/translated_instructions.jsonl"
    )
    parser.add_argument(
        "--coig-value",
        default="data/raw/modern/coig/human_value_alignment_part1.json",
    )
    parser.add_argument("--hsk", default="data/raw/lexicon/hsk/complete.json")
    parser.add_argument("--train-out", default="data/processed/modern_train_v3.txt")
    parser.add_argument("--val-out", default="data/processed/modern_val_v3.txt")
    parser.add_argument(
        "--manifest-out", default="data/processed/modern_mix_v3.manifest.json"
    )
    parser.add_argument("--seed", type=int, default=3307)
    parser.add_argument("--val-ratio", type=float, default=0.025)
    parser.add_argument("--coig-limit", type=int, default=10_000)
    parser.add_argument("--hsk-limit", type=int, default=4_000)
    args = parser.parse_args()
    manifest = build_modern_mix(
        legacy_path=Path(args.legacy),
        hc3_path=Path(args.hc3),
        coig_path=Path(args.coig),
        coig_value_path=Path(args.coig_value) if args.coig_value else None,
        hsk_path=Path(args.hsk),
        train_out=Path(args.train_out),
        val_out=Path(args.val_out),
        manifest_out=Path(args.manifest_out),
        seed=args.seed,
        val_ratio=args.val_ratio,
        coig_limit=args.coig_limit,
        hsk_limit=args.hsk_limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
