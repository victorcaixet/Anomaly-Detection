"""Marca uma caixa por imagem para o SAM; retoma de onde parou.

Uso: python scripts/annotate_boxes.py <pasta com train/ e test/> <boxes.json> [refazer.json]
Com refazer.json (lista de caminhos relativos), só remarca essas, partindo da caixa atual.
Arraste a caixa e ENTER confirma; 's' pula a imagem; Ctrl+C sai (salvo a cada caixa).
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("TkAgg")  # o OpenCV headless do projeto não abre janelas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.data import IMAGE_EXTENSIONS, list_files
from insulator_anomaly.preprocessing import read_image


def ask_box(image, title, initial=None):
    """Retorna [x0, y0, x1, y1] em px da imagem, ou None se pulada."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image[..., ::-1])
    ax.set_title(title)
    box = []
    selector = RectangleSelector(ax, lambda *_: None, interactive=True)
    if initial:
        x0, y0, x1, y1 = initial
        selector.extents = (x0, x1, y0, y1)
        selector.set_visible(True)

    def on_key(event):
        if event.key == "enter" and selector.extents[1] > selector.extents[0]:
            x0, x1, y0, y1 = selector.extents
            box.extend(round(v, 1) for v in (x0, y0, x1, y1))
            plt.close(fig)
        elif event.key == "s":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return box or None


def main(root, out, redo=None):
    root, out = Path(root).resolve(), Path(out)
    boxes = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    images = list_files(root, IMAGE_EXTENSIONS)
    redo = set(json.loads(Path(redo).read_text(encoding="utf-8"))) if redo else None
    for number, path in enumerate(images, 1):
        relative = path.relative_to(root).as_posix()
        if relative in boxes if redo is None else relative not in redo:
            continue
        box = ask_box(read_image(path), f"{number}/{len(images)} {relative}", boxes.get(relative))
        if box:
            boxes[relative] = box
            out.write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    print(f"{len(boxes)}/{len(images)} caixas em {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
