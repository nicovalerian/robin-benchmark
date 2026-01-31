import json
from pathlib import Path
from typing import Iterator
from datasets import load_dataset, Dataset


def load_jsonl(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def iter_jsonl(file_path: str | Path) -> Iterator[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_dataset_from_hub(
    dataset_name: str,
    split: str = "train",
    streaming: bool = False,
) -> Dataset:
    return load_dataset(dataset_name, split=split, streaming=streaming)
