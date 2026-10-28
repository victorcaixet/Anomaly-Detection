"""Clica pontos para corrigir máscaras SAM suspeitas; retoma de onde parou.

Uso: python scripts/annotate_points.py <pasta com train/ e test/> <boxes.json> <suspeitas.json> <points.json> [sam2.1_b.pt]
Botão esquerdo = parte do objeto, direito = fundo; 'z' desfaz; ENTER salva; 's' pula; Ctrl+C sai.
A máscara é recalculada a cada clique (caixa + pontos). Saída: {caminho: [[x, y, 1|0], ...]}.
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("TkAgg")  # o OpenCV headless do projeto não abre janelas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insulator_anomaly.preprocessing import read_image, sam_mask


def ask_points(sam, image, box, title, points):
    """Retorna a lista de pontos confirmada, ou None se pulada."""
    fig, ax = plt.subplots(figsize=(12, 8))
    result = []

    def draw():
        ax.clear()
        ax.imshow(image[..., ::-1])
        try:
            ax.imshow(sam_mask(sam, image, box=box, points=points), alpha=0.4, cmap="autumn")
        except RuntimeError:  # sem máscara: clique pontos até o SAM gerar uma
            pass
        ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec="cyan"))
        for x, y, label in points:
            ax.plot(x, y, "o", color="lime" if label else "red", mec="white")
        ax.set_title(title)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is ax and event.button in (1, 3) and not fig.canvas.toolbar.mode:
            points.append([round(event.xdata, 1), round(event.ydata, 1), int(event.button == 1)])
            draw()

    def on_key(event):
        if event.key == "z" and points:
            points.pop()
            draw()
        elif event.key == "enter":
            result.append(points)
            plt.close(fig)
        elif event.key == "s":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
    return result[0] if result else None


def main(root, boxes, flagged, out, model="sam2.1_b.pt"):
    import ultralytics

    root, out, sam = Path(root).resolve(), Path(out), ultralytics.SAM(model)
    boxes = json.loads(Path(boxes).read_text(encoding="utf-8"))
    flagged = json.loads(Path(flagged).read_text(encoding="utf-8"))
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for number, relative in enumerate(flagged, 1):
        if relative in done or relative not in boxes:
            continue
        points = ask_points(
            sam, read_image(root / relative), boxes[relative],
            f"{number}/{len(flagged)} {relative}: {'; '.join(flagged[relative])}", [],
        )
        if points:
            done[relative] = points
            out.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(f"{len(done)}/{len(flagged)} corrigidas em {out}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
