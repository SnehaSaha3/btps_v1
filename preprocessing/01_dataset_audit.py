from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter
import os

DATA_ROOT = Path("../data")

DATASETS = [
    "drug",
    "drug_train",
    "drug_valid",
    "handgun",
    "pistol",
    "knife",
    "knife_data",
    "knife_train",
    "knife_valid",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_label(image_path):
    """
    Assumes:
        image.jpg
        image.txt

    OR that images and labels are in corresponding
    train/images and train/labels directories.
    """

    # Normal same-folder annotation
    same_folder = image_path.with_suffix(".txt")

    if same_folder.exists():
        return same_folder

    # images/... -> labels/...
    parts = list(image_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        new_parts = parts[:]
        new_parts[idx] = "labels"

        label_path = Path(*new_parts).with_suffix(".txt")

        if label_path.exists():
            return label_path

    return None


def file_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def audit_dataset(name):

    folder = DATA_ROOT / name

    if not folder.exists():
        print(f"\n❌ {name}: folder not found")
        return

    images = get_images(folder)

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    print(f"Images found: {len(images)}")

    missing_labels = []
    corrupt_images = []
    invalid_boxes = []

    dimensions = Counter()
    class_counter = Counter()
    objects_per_image = []

    hashes = {}

    for image in images:

        # -------------------------
        # Image validation
        # -------------------------

        try:
            with Image.open(image) as img:
                width, height = img.size
                img.verify()

            dimensions[(width, height)] += 1

        except Exception:
            corrupt_images.append(image)
            continue

        # -------------------------
        # Duplicate detection
        # -------------------------

        try:
            h = file_hash(image)

            if h in hashes:
                hashes[h].append(image)
            else:
                hashes[h] = [image]

        except Exception:
            pass

        # -------------------------
        # Label validation
        # -------------------------

        label = find_label(image)

        if label is None:
            missing_labels.append(image)
            continue

        count = 0

        try:

            with open(label, "r") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, start=1):

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:
                    invalid_boxes.append(
                        (image, line_no, "Wrong number of values")
                    )
                    continue

                try:
                    class_id = int(values[0])

                    x = float(values[1])
                    y = float(values[2])
                    w = float(values[3])
                    h = float(values[4])

                except ValueError:
                    invalid_boxes.append(
                        (image, line_no, "Non-numeric annotation")
                    )
                    continue

                # YOLO normalized coordinates must be 0-1
                if not (
                    0 <= x <= 1 and
                    0 <= y <= 1 and
                    0 < w <= 1 and
                    0 < h <= 1
                ):
                    invalid_boxes.append(
                        (image, line_no, "Invalid bbox")
                    )
                    continue

                class_counter[class_id] += 1
                count += 1

        except Exception as e:

            invalid_boxes.append(
                (image, 0, str(e))
            )

        objects_per_image.append(count)

    # -------------------------
    # Duplicate groups
    # -------------------------

    duplicate_groups = [
        paths for paths in hashes.values()
        if len(paths) > 1
    ]

    # -------------------------
    # Results
    # -------------------------

    print("\nIMAGE STATISTICS")
    print("----------------")
    print(f"Valid images       : {len(images) - len(corrupt_images)}")
    print(f"Corrupt images     : {len(corrupt_images)}")
    print(f"Missing labels     : {len(missing_labels)}")
    print(f"Invalid annotations: {len(invalid_boxes)}")
    print(f"Duplicate groups   : {len(duplicate_groups)}")

    print("\nCLASS DISTRIBUTION")
    print("------------------")

    for cls, count in sorted(class_counter.items()):
        print(f"Class {cls}: {count} objects")

    print("\nIMAGE DIMENSIONS")
    print("----------------")

    for dimension, count in dimensions.most_common(10):
        print(f"{dimension}: {count}")

    if objects_per_image:
        print("\nOBJECTS PER IMAGE")
        print("-----------------")
        print(f"Minimum : {min(objects_per_image)}")
        print(f"Maximum : {max(objects_per_image)}")
        print(
            f"Average : "
            f"{sum(objects_per_image) / len(objects_per_image):.2f}"
        )

    if missing_labels:
        print("\n⚠️ MISSING LABEL EXAMPLES")

        for p in missing_labels[:10]:
            print(p)

    if corrupt_images:
        print("\n⚠️ CORRUPT IMAGE EXAMPLES")

        for p in corrupt_images[:10]:
            print(p)

    if invalid_boxes:
        print("\n⚠️ INVALID ANNOTATION EXAMPLES")

        for item in invalid_boxes[:10]:
            print(item)

    if duplicate_groups:
        print("\n⚠️ DUPLICATE IMAGE EXAMPLES")

        for group in duplicate_groups[:5]:
            print("\n".join(map(str, group)))


if __name__ == "__main__":

    for dataset in DATASETS:
        audit_dataset(dataset)