"""text5 benchmark (§4.1): AG News → Yelp → Amazon → DBpedia → Yahoo, 33 classes (Yelp/Amazon share labels)."""

from __future__ import annotations

import io
import tarfile
from typing import Callable

import pandas as pd
import torch
from torch import Tensor

from contilab.core.types import Stream, Task, TensorSplit
from contilab.data.cache import cached_tensors
from contilab.data.download import fetch
from contilab.data.splits import train_val_split

NLP = "https://s3.amazonaws.com/fast-ai-nlp"
WIKITEXT_URL = f"{NLP}/wikitext-2.tgz"
FULL_TRAIN, FULL_TEST = 115_000, 7_600  # original per-dataset sizes (de Masson d'Autume et al. 2019)


def _join(*cols: int) -> Callable[[pd.DataFrame], list[str]]:
    def fn(df: pd.DataFrame) -> list[str]:
        txt = df[cols[0]].fillna("").astype(str)
        for c in cols[1:]:
            txt = txt + " " + df[c].fillna("").astype(str)
        return [t.replace("\\n", " ").strip() for t in txt]

    return fn


# (archive url, text builder over CSV columns, global label offset, number of classes)
TEXT5_SOURCES: list[tuple[str, Callable[[pd.DataFrame], list[str]], int, int]] = [
    (f"{NLP}/ag_news_csv.tgz", _join(1, 2), 0, 4),
    (f"{NLP}/yelp_review_full_csv.tgz", _join(1), 4, 5),
    (f"{NLP}/amazon_review_full_csv.tgz", _join(1, 2), 4, 5),
    (f"{NLP}/dbpedia_csv.tgz", _join(1, 2), 9, 14),
    (f"{NLP}/yahoo_answers_csv.tgz", _join(1, 2, 3), 23, 10),
]


def _read_csv(url: str, split: str) -> pd.DataFrame:
    path = fetch(url)
    root = url.rsplit("/", 1)[-1].removesuffix(".tgz")
    with tarfile.open(path, "r:gz") as tf:
        data = tf.extractfile(f"{root}/{split}.csv").read()
    return pd.read_csv(io.BytesIO(data), header=None, dtype=str, keep_default_na=False)


def tokenize(texts: list[str], tokenizer_name: str, max_len: int) -> tuple[Tensor, Tensor]:
    """Right-padded token ids and attention mask; pad falls back to eos; empty texts become a single eos."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    texts = [t if t.strip() else tok.eos_token for t in texts]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    idx = torch.randperm(len(df), generator=torch.Generator().manual_seed(seed))[:n].sort().values
    return df.iloc[idx.numpy()]


def _task_tensors(url: str, fn, offset: int, tokenizer_name: str, n_train: int, n_test: int, max_len: int) -> dict[str, Tensor]:
    key = {"url": url, "tokenizer": tokenizer_name, "n_train": n_train, "n_test": n_test, "max_len": max_len}

    def build() -> dict[str, Tensor]:
        out = {}
        for split, n in (("train", n_train), ("test", n_test)):
            df = _sample(_read_csv(url, split), n, seed=0)
            ids, mask = tokenize(fn(df), tokenizer_name, max_len)
            out[f"{split}.x"], out[f"{split}.mask"] = ids, mask
            out[f"{split}.y"] = torch.tensor(df[0].astype(int).to_numpy() - 1 + offset, dtype=torch.long)
        return out

    return cached_tensors("text", key, build)


def wikitext_lines(split: str, n: int) -> list[str]:
    path = fetch(WIKITEXT_URL)
    with tarfile.open(path, "r:gz") as tf:
        df = pd.read_csv(io.BytesIO(tf.extractfile(f"wikitext-2/{split}.csv").read()), header=None, dtype=str)
    lines = []
    for article in df[0]:
        for ln in article.split("\n"):
            ln = ln.strip()
            if ln and not ln.startswith("="):
                lines.append(ln)
                if len(lines) == n:
                    return lines
    return lines


def wikitext_split(split: str, n: int, tokenizer_name: str, max_len: int) -> TensorSplit:
    key = {"url": WIKITEXT_URL, "split": split, "n": n, "tokenizer": tokenizer_name, "max_len": max_len}
    t = cached_tensors("text", key, lambda: dict(zip(("x", "mask"), tokenize(wikitext_lines(split, n), tokenizer_name, max_len))))
    return TensorSplit(t["x"], torch.zeros(t["x"].shape[0], dtype=torch.long), t["mask"])


def build_text5(tokenizer_name: str, scenario: str = "class", n_train: int | None = 2000, n_test: int | None = 500,
                max_len: int = 128, ppl_n: int = 200) -> Stream:
    n_train = FULL_TRAIN if n_train is None else n_train
    n_test = FULL_TEST if n_test is None else n_test
    tasks = []
    for t, (url, fn, offset, k) in enumerate(TEXT5_SOURCES):
        d = _task_tensors(url, fn, offset, tokenizer_name, n_train, n_test, max_len)
        train, val = train_val_split(TensorSplit(d["train.x"], d["train.y"], d["train.mask"]), 0.1, seed=t)
        test = TensorSplit(d["test.x"], d["test.y"], d["test.mask"])
        tasks.append(Task(t, train, val, test, torch.arange(offset, offset + k)))
    lm = wikitext_split("test", ppl_n, tokenizer_name, max_len)
    return Stream("text5", tasks, scenario, 33, "text", lm_eval=lm)
