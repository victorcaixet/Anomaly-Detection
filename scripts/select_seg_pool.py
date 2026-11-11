"""Seleciona o pool de segmentação: N imagens de train/good, em rajadas sem anomalia.

Uso: python scripts/select_seg_pool.py <dataset> <saida> [--images 300] [--bursts 100] [--sam 50] [--seed 42]
Rajada = mesma data e números DJI consecutivos (lacuna <= 5). Rajadas com test/anomaly
ficam reservadas (ouro/teste); o pool não compartilha rajada com elas.
Saída: pool.csv, bursts.csv e cópias em sam50/ e propagate/ (mesmos caminhos relativos).
"""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import random
import re
import shutil

PATTERN = re.compile(r"Fotos (\d\d-\d\d-\d{4})_DJI_(\d+)_")
GAP = 5


def find_bursts(root):
    """Retorna {burst_id: [(dji, relativo, split)]} e a lista de arquivos sem padrão."""
    by_date, unmatched = defaultdict(list), []
    for path in sorted(root.rglob("*.jpg")):
        relative = path.relative_to(root)
        match = PATTERN.match(path.name)
        if match:
            by_date[match[1]].append((int(match[2]), relative.as_posix(), relative.parent.as_posix()))
        else:
            unmatched.append(relative.as_posix())
    bursts = {}
    for date, items in by_date.items():
        items.sort()
        current = [items[0]]
        for item in items[1:]:
            if item[0] - current[-1][0] > GAP:
                bursts[f"{date}_{current[0][0]}"] = current
                current = []
            current.append(item)
        bursts[f"{date}_{current[0][0]}"] = current
    return bursts, unmatched


def largest_remainder(weights, total):
    """Divide total em cotas inteiras proporcionais a weights {chave: peso}."""
    scale = total / sum(weights.values())
    quota = {key: int(weight * scale) for key, weight in weights.items()}
    rest = sorted(weights, key=lambda key: weights[key] * scale - quota[key], reverse=True)
    for key in rest[: total - sum(quota.values())]:
        quota[key] += 1
    return quota


def spaced(frames, count):
    """count quadros uniformemente espaçados na ordem DJI."""
    if count >= len(frames):
        return list(frames)
    return [frames[round(i * (len(frames) - 1) / max(count - 1, 1))] for i in range(count)]


def select(bursts, images, n_bursts, n_sam, seed):
    rng = random.Random(seed)
    role = {}
    eligible = defaultdict(list)  # data -> rajadas elegíveis
    for burst_id, items in bursts.items():
        splits = {split for *_, split in items}
        if "test/anomaly" in splits:
            role[burst_id] = "reservada_anomalia"
        elif "train/good" in splits:
            role[burst_id] = "elegivel"
            eligible[burst_id.split("_")[0]].append(burst_id)
        else:
            role[burst_id] = "fora"
    quota = largest_remainder({d: len(b) for d, b in eligible.items()}, n_bursts)
    chosen = []
    for date, ids in sorted(eligible.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        chosen += ids[: quota[date]]
    train = {b: [i for i in bursts[b] if i[2] == "train/good"] for b in chosen}
    # Rodadas de um quadro por rajada até completar `images`.
    picked = {b: 0 for b in chosen}
    total = 0
    while total < images and any(picked[b] < len(train[b]) for b in chosen):
        for burst_id in chosen:
            if total < images and picked[burst_id] < len(train[burst_id]):
                picked[burst_id] += 1
                total += 1
    pool = {b: spaced(train[b], picked[b]) for b in chosen if picked[b]}
    # sam50: um quadro (o central) por rajada, proporcional por data.
    by_date = defaultdict(list)
    for burst_id in sorted(pool):
        by_date[burst_id.split("_")[0]].append(burst_id)
    sam_quota = largest_remainder({d: len(b) for d, b in by_date.items()}, n_sam)
    sam = set()
    for date, ids in sorted(by_date.items()):
        rng.shuffle(ids)
        sam |= {pool[b][len(pool[b]) // 2][1] for b in ids[: sam_quota[date]]}
    for burst_id in pool:
        role[burst_id] = "pool"
    return pool, sam, role


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--images", type=int, default=300)
    parser.add_argument("--bursts", type=int, default=100)
    parser.add_argument("--sam", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bursts, unmatched = find_bursts(args.dataset)
    pool, sam, role = select(bursts, args.images, args.bursts, args.sam, args.seed)
    args.output.mkdir(parents=True)
    with open(args.output / "pool.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "date", "burst_id", "subset"])
        for burst_id, items in sorted(pool.items()):
            for _, relative, _ in items:
                writer.writerow([relative, burst_id.split("_")[0], burst_id,
                                 "sam50" if relative in sam else "propagate"])
                target = args.output / ("sam50" if relative in sam else "propagate") / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.dataset / relative, target)
    with open(args.output / "bursts.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["burst_id", "role", "n_train_good", "n_test_good", "n_test_anomaly"])
        for burst_id, items in sorted(bursts.items()):
            count = lambda split: sum(1 for *_, s in items if s == split)
            writer.writerow([burst_id, role[burst_id], count("train/good"),
                             count("test/good"), count("test/anomaly")])
    print(f"Pool: {sum(map(len, pool.values()))} imagens em {len(pool)} rajadas; sam50: {len(sam)}.")
    print(f"Sem padrão de nome (fora do pool): {len(unmatched)}.")


if __name__ == "__main__":
    main()
