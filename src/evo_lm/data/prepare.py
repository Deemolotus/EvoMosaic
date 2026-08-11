"""Convert chinese-poetry JSON corpora into plain text + LaTeX books."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional


def _as_lines(obj: Any) -> List[str]:
    lines: List[str] = []
    if obj is None:
        return lines
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        for x in obj:
            lines.extend(_as_lines(x))
        return lines
    if isinstance(obj, dict):
        # common schemas in chinese-poetry
        for key in ("paragraphs", "content", "chapter", "title", "author", "rhythmic", "section"):
            if key in obj and key in {"paragraphs", "content"}:
                lines.extend(_as_lines(obj[key]))
            elif key in obj and key in {"title", "chapter", "author", "rhythmic", "section"} and isinstance(obj[key], str):
                # metadata handled by callers
                pass
        # nested guwenguanzhi-style
        if "content" in obj and isinstance(obj["content"], dict):
            lines.extend(_as_lines(obj["content"]))
        if "content" in obj and isinstance(obj["content"], list) and obj["content"] and isinstance(obj["content"][0], dict):
            for item in obj["content"]:
                lines.extend(_as_lines(item))
        return lines
    return lines


def extract_records(path: Path) -> Iterator[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        # Great Learning / Doctrine of the Mean style: single chapter dict
        if "paragraphs" in data:
            yield data
            return
        # Guwen Guanzhi style: {title, abstract, content: [...]}
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    yield item
                elif isinstance(item, str):
                    yield {"title": data.get("title", path.stem), "paragraphs": [item]}
            return
        yield data


def record_to_text(rec: dict[str, Any]) -> str:
    parts: List[str] = []
    title = rec.get("title") or rec.get("chapter") or rec.get("rhythmic")
    author = rec.get("author")
    section = rec.get("section")
    header_bits = [x for x in (section, title, author) if isinstance(x, str) and x]
    if header_bits:
        parts.append("·".join(header_bits))
    body = rec.get("paragraphs") or rec.get("content") or []
    if isinstance(body, str):
        parts.append(body)
    elif isinstance(body, list):
        for line in body:
            if isinstance(line, str):
                parts.append(line)
            elif isinstance(line, dict):
                # nested
                nested = record_to_text(line)
                if nested:
                    parts.append(nested)
    return "\n".join(p for p in parts if p).strip()


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def json_corpus_to_text(raw_dir: Path, out_txt: Path) -> int:
    files = sorted(raw_dir.rglob("*.json"))
    n = 0
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as fout:
        for fp in files:
            try:
                for rec in extract_records(fp):
                    text = record_to_text(rec)
                    if not text:
                        continue
                    fout.write(text)
                    fout.write("\n\n")
                    n += 1
            except Exception as e:
                print(f"[warn] skip {fp}: {e}")
    return n


def json_to_latex_book(
    json_path: Path,
    out_tex: Path,
    *,
    title: Optional[str] = None,
    author: str = "Classical corpus (compiled)",
    max_records: int = 500,
) -> Path:
    """Emit a XeLaTeX/LuaLaTeX-friendly book for one corpus file."""
    title = title or json_path.stem
    records = list(extract_records(json_path))[:max_records]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\documentclass[12pt,a4paper]{ctexbook}",
        r"\usepackage{geometry}",
        r"\geometry{margin=2.2cm}",
        r"\usepackage{setspace}",
        r"\setstretch{1.35}",
        r"\usepackage{hyperref}",
        rf"\title{{{latex_escape(title)}}}",
        rf"\author{{{latex_escape(author)}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
    ]
    for i, rec in enumerate(records, 1):
        heading = rec.get("title") or rec.get("chapter") or rec.get("rhythmic") or f"第{i}则"
        auth = rec.get("author")
        sec = rec.get("section")
        lines.append(rf"\section{{{latex_escape(str(heading))}}}")
        if sec or auth:
            meta = "　".join(x for x in (sec, auth) if x)
            lines.append(rf"\textbf{{{latex_escape(meta)}}}\par\vspace{{0.5em}}")
        body = rec.get("paragraphs") or rec.get("content") or []
        if isinstance(body, str):
            body = [body]
        for para in body:
            if isinstance(para, str) and para.strip():
                lines.append(latex_escape(para.strip()) + r"\par")
            elif isinstance(para, dict):
                t = record_to_text(para)
                if t:
                    for p in t.split("\n"):
                        lines.append(latex_escape(p) + r"\par")
        lines.append("")
    lines.append(r"\end{document}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    return out_tex


def export_all_latex(raw_dir: Path, latex_dir: Path, max_records: int = 400) -> List[Path]:
    """Batch-export curated classical JSON files to XeLaTeX books.

    Mapping values stay Chinese (canonical work titles for the PDF catalog).
    """
    latex_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    mapping = {
        "lunyu": "论语",
        "shijing": "诗经",
        "daxue": "大学",
        "zhongyong": "中庸",
        "mengzi": "孟子",
        "chuci": "楚辞",
        "guwenguanzhi": "古文观止",
        "sanzijing": "三字经",
        "qianziwen": "千字文",
        "baijiaxing": "百家姓",
        "dizigui": "弟子规",
        "youmengying": "幽梦影",
        "caocao": "曹操诗集",
        "songci300": "宋词三百首",
        "tangshisanbaishou": "唐诗三百首",
        "nalanxingde": "纳兰性德诗集",
        "yuanqu": "元曲选辑",
        "zengguangxianwen": "增广贤文",
        "youxueqionglin": "幼学琼林",
        "qianjiashi": "千家诗",
        "shenglvqimeng": "声律启蒙",
    }
    for fp in sorted(raw_dir.rglob("*.json")):
        stem = fp.stem
        # skip huge shard dumps for latex book (still in training text)
        if stem.startswith(("ci.song.", "poet.tang.", "poet.song.", "huajianji-")):
            # still export a short sampler
            title = {
                True: "花间集选",
            }.get(stem.startswith("huajianji"), None)
            if stem.startswith("ci.song."):
                title = f"全宋词选·{stem}"
            elif stem.startswith("poet.tang."):
                title = f"全唐诗选·{stem}"
            elif stem.startswith("poet.song."):
                title = f"全宋诗选·{stem}"
            elif stem.startswith("huajianji"):
                title = f"花间集·{stem}"
            out = latex_dir / f"{stem}.tex"
            json_to_latex_book(fp, out, title=title or stem, max_records=min(80, max_records))
            written.append(out)
            continue
        title = mapping.get(stem, stem)
        out = latex_dir / f"{stem}.tex"
        lim = 200 if stem == "yuanqu" else max_records
        json_to_latex_book(fp, out, title=title, max_records=lim)
        written.append(out)
    # Master index (Chinese filenames kept for the Classical LaTeX catalog)
    index = latex_dir / "00_著作总目.tex"
    body = [
        r"\documentclass[12pt]{ctexart}",
        r"\begin{document}",
        r"\section*{EvoLM Classical Corpus LaTeX Index}",
        r"Auto-generated by the data pipeline. Full texts are sibling \texttt{.tex} files.",
        r"\begin{enumerate}",
    ]
    for p in written:
        body.append(rf"\item {latex_escape(p.name)}")
    body += [r"\end{enumerate}", r"\end{document}"]
    index.write_text("\n".join(body), encoding="utf-8")
    written.insert(0, index)
    return written


# Seed dialogues for the legacy classical+chat corpus (Chinese protocol text).
DIALOGUE_SEED = """用户: 你好，今天天气怎么样？
助手: 安好。窗外疏雨微凉，正宜静读一卷诗书。

用户: 你能给我讲讲论语里学而时习之的意思吗？
助手: 「学而时习之，不亦说乎」——学了知识又按时温习，心中自有喜悦。这是把求知当作日用常行，而非一曝十寒。

用户: 帮我写一首关于春天的短诗。
助手: 东风潜入柳，细雨润无声。
燕雀衔泥急，闲花落满城。

用户: 什么是进化算法？
助手: 进化算法模仿自然选择：先有一群「候选解」（种群），按适应度择优，再通过交叉与变异产生后代，一代代逼近更好的结构或参数。在本项目里，种群里的每个个体是一套混合网络蓝图。

用户: 红楼梦是谁写的？
助手: 曹雪芹著，后四十回一般认为由高鹗续成。虽是白话章回小说，其诗词典故仍与古典传统一脉相承。

用户: 晚上吃什么好？
助手: 若求清淡，蔬粥一盏即可；若欲对酒当歌，小酌慢饮，勿过饱。

用户: 用文言文回复：今天有点累。
助手: 劳形者日昃而倦，宜少息。烹茶一盏，闭目片时，神气自复。

用户: 孟子见梁惠王讲了什么？
助手: 开篇即问「亦将有以利吾国乎」，孟子答以「何必曰利？亦有仁义而已矣」，把治国从功利拉回到仁义根基。

用户: 给我一句诗经里的离别。
助手: 「昔我往矣，杨柳依依；今我来思，雨雪霏霏。」——《小雅·采薇》。

用户: 你会聊天吗？
助手: 会。可谈日常起居，亦可共读诗书。你想从哪一话题起？
"""


def write_dialogue_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DIALOGUE_SEED.strip() + "\n", encoding="utf-8")
