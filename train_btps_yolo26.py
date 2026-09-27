from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "combined_dataset" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

EPOCHS = 40
IMAGE_SIZE = 640
BATCH_SIZE = 16
SEED = 42


def get_device():
    if torch.cuda.is_available():
        print("Device: CUDA GPU")
        return 0

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Device: Apple MPS")
        return "mps"

    print("Device: CPU")
    return "cpu"


def main():
    print("=" * 70)
    print("BTPS_V1 - YOLO26 TRAINING")
    print("=" * 70)
    print(f"Dataset : {DATA_YAML}")
    print(f"Epochs  : {EPOCHS}")
    print(f"Image   : {IMAGE_SIZE}")
    print(f"Batch   : {BATCH_SIZE}")
    print()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found:\n{DATA_YAML}")

    device = get_device()

    # YOLO26 nano is the first training run to keep compute/time manageable.
    model = YOLO("yolo26n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        seed=SEED,
        val=True,
        save=True,
        save_period=5,
        cache=False,
        workers=4,
        patience=12,
        project=str(RUNS_DIR),
        name="btps_yolo26n_4class",
        plots=True,
        amp=True,
        mosaic=1.0,
        close_mosaic=10,
        verbose=True,
    )

    run_dir = RUNS_DIR / "btps_yolo26n_4class"

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Best weights : {run_dir / 'weights' / 'best.pt'}")
    print(f"Last weights : {run_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
