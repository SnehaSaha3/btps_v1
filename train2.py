from pathlib import Path

# ============================================================
# BTPS - YOLO26 TRAINING 2
# Completely separate experiment
# ============================================================

SCRIPT_DIR = Path("/content/btps_v1")
DRIVE_ROOT = Path("/content/drive/MyDrive/btps_v1")

RUNS_DIR = DRIVE_ROOT / "runs"
RUN_DIR = RUNS_DIR / "btps_yolo26n_training2"

RUN_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_PATH = DRIVE_ROOT / "train_btps_yolo26_training2.py"

script = r'''
from pathlib import Path
from ultralytics import YOLO
import torch
import json
import shutil


# ============================================================
# BTPS YOLO26 - TRAINING 2
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/btps_v1")

DATA_YAML = Path("/content/combined_dataset/data.yaml")

RUNS_DIR = DRIVE_ROOT / "runs"
RUN_NAME = "btps_yolo26n_training2"
RUN_DIR = RUNS_DIR / RUN_NAME

BEST_MODEL = RUN_DIR / "weights" / "best.pt"
LAST_MODEL = RUN_DIR / "weights" / "last.pt"

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL_NAME = "yolo26n.pt"

EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 16

DEVICE = 0
WORKERS = 2

SEED = 42
PATIENCE = 20

# ============================================================
# YOLO26n OPTIMIZATION
# Based on the documented YOLO26n recipe
# ============================================================

OPTIMIZER = "MuSGD"

LR0 = 0.0054
LRF = 0.0495

MOMENTUM = 0.947
WEIGHT_DECAY = 0.00064

WARMUP_EPOCHS = 0.98

# ============================================================
# YOLO26n LOSS WEIGHTS
# ============================================================

BOX = 5.63
CLS = 0.56
DFL = 9.04

# ============================================================
# AUGMENTATION
# ============================================================

MOSAIC = 0.909
MIXUP = 0.012
COPY_PASTE = 0.075

SCALE = 0.562

FLIPLR = 0.606
FLIPUD = 0.0

DEGREES = 1.11
SHEAR = 1.46
TRANSLATE = 0.071
PERSPECTIVE = 0.0

HSV_H = 0.014
HSV_S = 0.645
HSV_V = 0.566

BGR = 0.106

CLOSE_MOSAIC = 10

# ============================================================
# OTHER TRAINING SETTINGS
# ============================================================

AMP = True
VAL = True
PLOTS = True
SAVE = True
SAVE_PERIOD = 1
EXIST_OK = True

DETERMINISTIC = True
VERBOSE = True


# ============================================================
# GPU CHECK
# ============================================================

def check_gpu():

    print("=" * 70)
    print("GPU CHECK")
    print("=" * 70)

    if torch.cuda.is_available():

        print("CUDA available: YES")
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB"
        )

    else:

        print("WARNING: CUDA is NOT available.")


# ============================================================
# DATASET CHECK
# ============================================================

def check_dataset():

    print("\n" + "=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    print("Dataset YAML:", DATA_YAML)
    print("Exists:", DATA_YAML.exists())

    required_dirs = [

        Path("/content/combined_dataset/images/train"),
        Path("/content/combined_dataset/images/valid"),
        Path("/content/combined_dataset/images/test"),

        Path("/content/combined_dataset/labels/train"),
        Path("/content/combined_dataset/labels/valid"),
        Path("/content/combined_dataset/labels/test"),

    ]

    for directory in required_dirs:

        print(
            f"{directory}:",
            "OK" if directory.exists() else "MISSING"
        )

        if not directory.exists():
            raise FileNotFoundError(
                f"Required dataset directory missing: {directory}"
            )

    print("\nDATA YAML")
    print("-" * 70)
    print(DATA_YAML.read_text())


# ============================================================
# SAVE EXPERIMENT CONFIGURATION
# ============================================================

def save_config():

    config = {

        "experiment": "BTPS YOLO26 Training 2",

        "model": MODEL_NAME,

        "dataset": str(DATA_YAML),

        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,

        "device": DEVICE,
        "workers": WORKERS,

        "seed": SEED,
        "patience": PATIENCE,

        "optimizer": OPTIMIZER,

        "lr0": LR0,
        "lrf": LRF,

        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,

        "warmup_epochs": WARMUP_EPOCHS,

        "box": BOX,
        "cls": CLS,
        "dfl": DFL,

        "mosaic": MOSAIC,
        "mixup": MIXUP,
        "copy_paste": COPY_PASTE,

        "scale": SCALE,

        "fliplr": FLIPLR,
        "flipud": FLIPUD,

        "degrees": DEGREES,
        "shear": SHEAR,
        "translate": TRANSLATE,
        "perspective": PERSPECTIVE,

        "hsv_h": HSV_H,
        "hsv_s": HSV_S,
        "hsv_v": HSV_V,

        "bgr": BGR,

        "close_mosaic": CLOSE_MOSAIC,

        "amp": AMP,
        "val": VAL,
        "plots": PLOTS,

        "save": SAVE,
        "save_period": SAVE_PERIOD,

        "deterministic": DETERMINISTIC,

        "run_directory": str(RUN_DIR),

    }

    config_path = RUN_DIR / "training2_config.json"

    with open(config_path, "w") as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print("\nConfiguration saved to:")
    print(config_path)


# ============================================================
# TRAIN
# ============================================================

def train():

    print("\n" + "=" * 70)
    print("STARTING BTPS YOLO26 TRAINING 2")
    print("=" * 70)

    print("Run directory:")
    print(RUN_DIR)

    print("\nLoading:", MODEL_NAME)

    model = YOLO(MODEL_NAME)

    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,

        optimizer=OPTIMIZER,

        lr0=LR0,
        lrf=LRF,

        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,

        warmup_epochs=WARMUP_EPOCHS,

        box=BOX,
        cls=CLS,
        dfl=DFL,

        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,

        scale=SCALE,

        fliplr=FLIPLR,
        flipud=FLIPUD,

        degrees=DEGREES,
        shear=SHEAR,
        translate=TRANSLATE,
        perspective=PERSPECTIVE,

        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,

        bgr=BGR,

        close_mosaic=CLOSE_MOSAIC,

        seed=SEED,
        deterministic=DETERMINISTIC,

        patience=PATIENCE,

        workers=WORKERS,

        amp=AMP,

        val=VAL,
        plots=PLOTS,

        save=SAVE,
        save_period=SAVE_PERIOD,

        project=str(RUNS_DIR),
        name=RUN_NAME,

        exist_ok=EXIST_OK,

        verbose=VERBOSE,
    )

    return results


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test():

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    if not BEST_MODEL.exists():

        print("ERROR: best.pt was not found.")
        print(BEST_MODEL)

        return

    print("Loading best model:")
    print(BEST_MODEL)

    model = YOLO(str(BEST_MODEL))

    test_dir = RUN_DIR / "test_evaluation"

    metrics = model.val(

        data=str(DATA_YAML),

        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,

        plots=True,

        project=str(RUN_DIR),
        name="test_evaluation",

        exist_ok=True,

        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TEST METRICS")
    print("=" * 70)

    try:

        print(
            "Precision:",
            metrics.box.mp
        )

        print(
            "Recall:",
            metrics.box.mr
        )

        print(
            "mAP50:",
            metrics.box.map50
        )

        print(
            "mAP50-95:",
            metrics.box.map
        )

    except Exception as e:

        print("Could not print summary metrics:", e)


# ============================================================
# VERIFY RESULTS
# ============================================================

def verify_results():

    print("\n" + "=" * 70)
    print("TRAINING 2 RESULT VERIFICATION")
    print("=" * 70)

    important_files = [

        RUN_DIR / "results.csv",
        RUN_DIR / "results.png",

        RUN_DIR / "confusion_matrix.png",
        RUN_DIR / "confusion_matrix_normalized.png",

        RUN_DIR / "PR_curve.png",
        RUN_DIR / "P_curve.png",
        RUN_DIR / "R_curve.png",
        RUN_DIR / "F1_curve.png",

        BEST_MODEL,
        LAST_MODEL,

    ]

    for path in important_files:

        if path.exists():

            print("FOUND :", path)

        else:

            print("MISSING:", path)

    print("\n" + "=" * 70)
    print("TRAINING 2 DIRECTORY")
    print("=" * 70)

    if RUN_DIR.exists():

        for path in sorted(RUN_DIR.rglob("*")):

            if path.is_file():

                size_mb = path.stat().st_size / (1024**2)

                print(
                    f"{size_mb:8.2f} MB  "
                    f"{path.relative_to(RUN_DIR)}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    check_gpu()

    check_dataset()

    save_config()

    train()

    evaluate_test()

    verify_results()

    print("\n" + "=" * 70)
    print("BTPS YOLO26 TRAINING 2 FINISHED")
    print("=" * 70)

    print("\nBest model:")
    print(BEST_MODEL)

    print("\nLast checkpoint:")
    print(LAST_MODEL)

    print("\nRun directory:")
    print(RUN_DIR)


if __name__ == "__main__":

    main()
'''

SCRIPT_PATH.write_text(script)

print("=" * 70)
print("TRAINING 2 SCRIPT CREATED")
print("=" * 70)
print(SCRIPT_PATH)
print("Exists:", SCRIPT_PATH.exists())