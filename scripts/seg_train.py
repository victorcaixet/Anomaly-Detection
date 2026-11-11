"""Dataset final e treino dos segmentadores.

  run <review.csv> <pool.csv> <dataset> <saida> [--models yolo unet segformer] [--epochs 60]
      Tudo de uma vez: finaliza as máscaras revisadas em <saida>/data e treina YOLO26, U-Net e
      SegFormer em sequência (um começa quando o outro termina). Se rodar de novo na mesma
      <saida>, pula os já concluídos; --retry apaga e refaz os incompletos.
  finalize <review.csv> <pool.csv> <dataset> <saida> [--val 0.2]
      Só com status aceita/corrigida/refeita. Divide por rajada em train/val.
  train <data_dir> <saida> [--models ...] [--epochs 60] [--retry]
      Só o treino, sobre um dataset já finalizado.
  evaluate <kind> <weights> <split_dir> <saida.csv>
      Avalia um modelo treinado em outra pasta images/ + masks/ (ex.: o conjunto ouro).
Cada modelo grava <saida>/<modelo>/summary.json; comparison.csv junta os concluídos (IoU, Dice,
falhas, ms/img, parâmetros). Falha em um modelo vai para <modelo>_erro.txt e não para os outros.
Use o conjunto ouro (manual, em rajadas disjuntas) para os números do artigo: val/ também
escolhe o checkpoint, então seu IoU é otimista.
"""

from pathlib import Path
import argparse
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.seg_data import finalize


def run_finalize(args):
    rows = finalize(args.review, args.pool, args.dataset, args.output, args.val)
    split = [r["split"] for r in rows]
    print(f"train: {split.count('train')}, val: {split.count('val')}")


def train_and_report(args, data_dir, output):
    from insulator_anomaly.seg_models import train_models

    results, errors = train_models(args.models, data_dir, output, args.epochs, args.seed,
                                   args.yolo_weights, args.encoder, args.segformer, args.retry)
    for row in results:
        print(row["model"], {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in row.items() if k not in ("model", "weights")})
    if errors:
        raise SystemExit("Falharam: " + ", ".join(errors) + f". Veja <modelo>_erro.txt em {output}.")


def run_train(args):
    train_and_report(args, args.data_dir, args.output)


def run_all(args):
    output = Path(args.output)
    data = output / "data"
    digest = hashlib.sha256(Path(args.review).read_bytes()).hexdigest()
    if data.exists():  # retomada: o review.csv não pode ter mudado
        if (output / "review.sha256").read_text(encoding="utf-8") != digest:
            raise SystemExit("review.csv mudou desde a finalização: use outra pasta de saída.")
        print(f"Reaproveitando {data}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        rows = finalize(args.review, args.pool, args.dataset, data, args.val)
        (output / "review.sha256").write_text(digest, encoding="utf-8")
        split = [r["split"] for r in rows]
        print(f"train: {split.count('train')}, val: {split.count('val')}")
    train_and_report(args, data, output)


def run_evaluate(args):
    from insulator_anomaly import seg_models as sm

    device = "cuda" if sm.torch.cuda.is_available() else "cpu"
    _, summary = sm.evaluate(sm.load_predictor(args.kind, args.weights, device), args.split_dir, args.output)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()})


def add_training_options(parser):
    parser.add_argument("--models", nargs="+", default=["yolo", "unet", "segformer"],
                        choices=["yolo", "unet", "segformer"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-weights", default="yolo26s-seg.pt")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--segformer", default="nvidia/mit-b1")
    parser.add_argument("--retry", action="store_true")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    everything = sub.add_parser("run")
    for name in ("review", "pool", "dataset", "output"):
        everything.add_argument(name)
    everything.add_argument("--val", type=float, default=0.2)
    add_training_options(everything)
    everything.set_defaults(run=run_all)
    final = sub.add_parser("finalize")
    for name in ("review", "pool", "dataset", "output"):
        final.add_argument(name)
    final.add_argument("--val", type=float, default=0.2)
    final.set_defaults(run=run_finalize)
    train = sub.add_parser("train")
    train.add_argument("data_dir")
    train.add_argument("output")
    add_training_options(train)
    train.set_defaults(run=run_train)
    evaluate = sub.add_parser("evaluate")
    for name in ("kind", "weights", "split_dir", "output"):
        evaluate.add_argument(name)
    evaluate.set_defaults(run=run_evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
