# combine_and_split.py
#
# BTPS_V1
# Combines:
#   0 = drug
#   1 = handgun
#   2 = pistol
#   3 = knife
#
# Converts YOLO + COCO annotations to one YOLO dataset.
# Removes exact duplicate images.
# Splits at IMAGE level:
#   80% train
#   10% valid
#   10% test
#
# Output:
# data/combined_dataset/
# ├── images/
# │   ├── train/
# │   ├── valid/
# │   └── test/
# ├── labels/
# │   ├── train/
# │   ├── valid/
# │   └── test/
# ├── data.yaml
# └── split_manifest.csv
#
# Original datasets are NOT modified.

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random
import shutil
import csv
import re

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_ROOT = DATA_ROOT / "combined_dataset"

SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

CLASS_MAP = {
    "drug": 0,
    "handgun": 1,
    "pistol": 2,
    "knife": 3,
}

CLASS_NAMES = {
    0: "drug",
    1: "handgun",
    2: "pistol",
    3: "knife",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# DATASET SOURCE DETECTION
# ============================================================

def detect_class(path: Path):

    text = str(path).lower().replace("\\", "/")

    # IMPORTANT:
    # Check pistol before "pistol" can accidentally be confused
    # with another path.
    if "pistol" in text:
        return "pistol"

    if "handgun" in text:
        return "handgun"

    if "drug detection" in text or "/drug/" in text or "drug_" in text:
        return "drug"

    if "knife" in text or "knives" in text:
        return "knife"

    return None


# ============================================================
# HELPERS
# ============================================================

def is_image(path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_stem(filename):
    return Path(filename).stem.lower().strip()


def md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_name(name):

    name = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        name
    )

    return name


def make_unique_name(image, class_name):

    digest = md5(image)[:10]

    stem = safe_name(image.stem)

    extension = image.suffix.lower()

    return f"{class_name}__{stem}__{digest}{extension}"


# ============================================================
# COCO
# ============================================================

def load_coco_files():

    coco_files = []

    for json_file in DATA_ROOT.rglob("*.json"):

        if OUTPUT_ROOT in json_file.parents:
            continue

        try:

            with open(
                json_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if (
                isinstance(data, dict)
                and "images" in data
                and "annotations" in data
            ):
                coco_files.append(
                    (json_file, data)
                )

        except Exception:
            pass

    return coco_files


def build_coco_index():

    index = []

    for json_path, data in load_coco_files():

        class_name = detect_class(json_path)

        if class_name is None:
            continue

        image_index = {}

        for image in data.get("images", []):

            image_id = image.get("id")

            file_name = image.get("file_name")

            if image_id is None or file_name is None:
                continue

            image_index[image_id] = {
                "file_name": str(file_name),
                "width": image.get("width"),
                "height": image.get("height"),
            }

        annotations = defaultdict(list)

        for ann in data.get("annotations", []):

            image_id = ann.get("image_id")

            bbox = ann.get("bbox")

            if image_id is None:
                continue

            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                continue

            annotations[image_id].append(bbox)

        index.append({
            "json": json_path,
            "class_name": class_name,
            "images": image_index,
            "annotations": annotations,
        })

    return index


def find_coco_image(image_path, coco_index):

    stem = normalize_stem(image_path.name)

    matches = []

    for dataset in coco_index:

        for image_id, info in dataset["images"].items():

            coco_stem = normalize_stem(
                info["file_name"]
            )

            if coco_stem == stem:

                matches.append(
                    (
                        dataset,
                        image_id,
                        info
                    )
                )

    if not matches:
        return None

    # Prefer COCO JSON located closest to image.
    matches.sort(
        key=lambda x: abs(
            len(
                image_path.parts
            )
            -
            len(
                x[0]["json"].parts
            )
        )
    )

    return matches[0]


# ============================================================
# YOLO LABEL SEARCH
# ============================================================

def build_yolo_index():

    index = defaultdict(list)

    for txt in DATA_ROOT.rglob("*.txt"):

        if OUTPUT_ROOT in txt.parents:
            continue

        filename = txt.name.lower()

        if filename.startswith("readme"):
            continue

        stem = normalize_stem(
            txt.name
        )

        index[stem].append(txt)

    return index


def find_yolo_label(image, yolo_index):

    stem = normalize_stem(
        image.name
    )

    candidates = yolo_index.get(
        stem,
        []
    )

    if not candidates:
        return None

    # Same directory
    for candidate in candidates:

        if candidate.parent == image.parent:
            return candidate

    # images/ -> labels/
    if image.parent.name.lower() == "images":

        for candidate in candidates:

            if (
                candidate.parent.name.lower()
                == "labels"
                and candidate.parent.parent
                == image.parent.parent
            ):
                return candidate

    # Search ancestors
    ancestor = image.parent

    for _ in range(5):

        labels_dir = ancestor / "labels"

        if labels_dir.exists():

            for candidate in candidates:

                if candidate.parent == labels_dir:
                    return candidate

        if ancestor.parent == ancestor:
            break

        ancestor = ancestor.parent

    # Exact unique candidate
    if len(candidates) == 1:
        return candidates[0]

    return None


# ============================================================
# YOLO PARSER
# ============================================================

def read_yolo_label(
    label_file,
    target_class
):

    result = []

    try:

        lines = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    except Exception:

        return result

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:

            # Original class is intentionally ignored.
            #
            # The source dataset determines the final BTPS
            # class because these are separate source datasets.
            #
            # Example:
            # all YOLO objects inside the drug dataset
            # become class 0.
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:

            continue

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            continue

        result.append(
            (
                CLASS_MAP[target_class],
                xc,
                yc,
                w,
                h
            )
        )

    return result


# ============================================================
# COCO CONVERSION
# ============================================================

def read_coco_annotations(
    image_path,
    coco_match,
    target_class
):

    if coco_match is None:
        return []

    dataset, image_id, info = coco_match

    width = info.get("width")
    height = info.get("height")

    if not width or not height:

        try:

            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size

        except Exception:

            return []

    if width <= 0 or height <= 0:
        return []

    result = []

    for bbox in dataset["annotations"].get(
        image_id,
        []
    ):

        x, y, bw, bh = map(
            float,
            bbox
        )

        if bw <= 0 or bh <= 0:
            continue

        # COCO:
        # x, y, width, height
        #
        # YOLO:
        # center_x, center_y, width, height

        xc = (
            x + bw / 2
        ) / width

        yc = (
            y + bh / 2
        ) / height

        nw = bw / width
        nh = bh / height

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < nw <= 1
            and 0 < nh <= 1
        ):
            continue

        result.append(
            (
                CLASS_MAP[target_class],
                xc,
                yc,
                nw,
                nh
            )
        )

    return result


# ============================================================
# FIND IMAGES
# ============================================================

def collect_images():

    images = []

    for image in DATA_ROOT.rglob("*"):

        if not is_image(image):
            continue

        if OUTPUT_ROOT in image.parents:
            continue

        class_name = detect_class(image)

        if class_name is None:
            continue

        images.append(
            (
                image,
                class_name
            )
        )

    return images


# ============================================================
# ANNOTATION EXTRACTION
# ============================================================

def get_annotations(
    image,
    class_name,
    yolo_index,
    coco_index
):

    # --------------------------------------------------------
    # Try YOLO first
    # --------------------------------------------------------

    label_file = find_yolo_label(
        image,
        yolo_index
    )

    if label_file is not None:

        annotations = read_yolo_label(
            label_file,
            class_name
        )

        if annotations:

            return annotations, "YOLO"

    # --------------------------------------------------------
    # Try COCO
    # --------------------------------------------------------

    coco_match = find_coco_image(
        image,
        coco_index
    )

    if coco_match is not None:

        annotations = read_coco_annotations(
            image,
            coco_match,
            class_name
        )

        if annotations:

            return annotations, "COCO"

    return [], "NONE"


# ============================================================
# STRATIFIED IMAGE SPLIT
# ============================================================

def split_images(records):

    random.seed(SEED)

    groups = defaultdict(list)

    # Group by class.
    #
    # This ensures every class is represented in train/valid/test
    # as much as possible.
    for record in records:

        groups[
            record["class_name"]
        ].append(record)

    train = []
    valid = []
    test = []

    for class_name, items in groups.items():

        random.shuffle(items)

        n = len(items)

        train_count = int(
            n * TRAIN_RATIO
        )

        valid_count = int(
            n * VALID_RATIO
        )

        train.extend(
            items[:train_count]
        )

        valid.extend(
            items[
                train_count:
                train_count + valid_count
            ]
        )

        test.extend(
            items[
                train_count + valid_count:
            ]
        )

    # Shuffle final splits.
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    return train, valid, test


# ============================================================
# WRITE DATASET
# ============================================================

def create_directories():

    if OUTPUT_ROOT.exists():

        print(
            f"Removing previous combined dataset:\n"
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in [
        "train",
        "valid",
        "test"
    ]:

        (
            OUTPUT_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


def write_data_yaml():

    yaml_text = """path: .

train: images/train
val: images/valid
test: images/test

nc: 4

names:
  0: drug
  1: handgun
  2: pistol
  3: knife
"""

    (
        OUTPUT_ROOT / "data.yaml"
    ).write_text(
        yaml_text,
        encoding="utf-8"
    )


def write_split(
    records,
    split,
    manifest
):

    image_dir = (
        OUTPUT_ROOT
        / "images"
        / split
    )

    label_dir = (
        OUTPUT_ROOT
        / "labels"
        / split
    )

    for record in records:

        source_image = record[
            "image"
        ]

        class_name = record[
            "class_name"
        ]

        new_image_name = record[
            "output_name"
        ]

        new_label_name = (
            Path(new_image_name).stem
            + ".txt"
        )

        destination_image = (
            image_dir
            / new_image_name
        )

        destination_label = (
            label_dir
            / new_label_name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        with open(
            destination_label,
            "w",
            encoding="utf-8"
        ) as f:

            for annotation in record[
                "annotations"
            ]:

                cls, xc, yc, w, h = annotation

                f.write(
                    f"{cls} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}\n"
                )

        manifest.append({
            "split": split,
            "class": class_name,
            "class_id": CLASS_MAP[
                class_name
            ],
            "source_image": str(
                source_image.relative_to(
                    DATA_ROOT
                )
            ),
            "output_image": str(
                Path("images")
                / split
                / new_image_name
            ),
            "output_label": str(
                Path("labels")
                / split
                / new_label_name
            ),
            "annotation_type": record[
                "annotation_type"
            ],
            "object_count": len(
                record["annotations"]
            ),
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BTPS_V1 - COMBINE + CLEAN + SPLIT")
    print("=" * 70)

    print()
    print("Classes:")
    print("0 = drug")
    print("1 = handgun")
    print("2 = pistol")
    print("3 = knife")

    print()
    print("Split:")
    print("80% = train")
    print("10% = valid")
    print("10% = test")

    print()
    print("Searching datasets...")

    yolo_index = build_yolo_index()

    coco_index = build_coco_index()

    images = collect_images()

    print(
        f"Images discovered: {len(images)}"
    )

    print(
        f"YOLO label files discovered: "
        f"{len(yolo_index)}"
    )

    print(
        f"COCO datasets discovered: "
        f"{len(coco_index)}"
    )

    # --------------------------------------------------------
    # Build clean records
    # --------------------------------------------------------

    records = []

    seen_hashes = {}

    skipped_duplicate = 0
    skipped_no_label = 0
    skipped_invalid = 0

    for image, class_name in images:

        try:

            image_hash = md5(image)

        except Exception:

            skipped_invalid += 1
            continue

        # Exact duplicate protection.
        if image_hash in seen_hashes:

            skipped_duplicate += 1
            continue

        annotations, annotation_type = get_annotations(
            image,
            class_name,
            yolo_index,
            coco_index
        )

        # Do not put unlabeled images into the
        # object-detection training dataset.
        if not annotations:

            skipped_no_label += 1
            continue

        seen_hashes[
            image_hash
        ] = image

        output_name = make_unique_name(
            image,
            class_name
        )

        records.append({
            "image": image,
            "class_name": class_name,
            "annotations": annotations,
            "annotation_type": annotation_type,
            "output_name": output_name,
            "hash": image_hash,
        })

    print()
    print(
        f"Usable annotated images: "
        f"{len(records)}"
    )

    print(
        f"Duplicate images skipped: "
        f"{skipped_duplicate}"
    )

    print(
        f"Images without usable labels skipped: "
        f"{skipped_no_label}"
    )

    print(
        f"Invalid images skipped: "
        f"{skipped_invalid}"
    )

    # --------------------------------------------------------
    # Class counts BEFORE split
    # --------------------------------------------------------

    class_counts = defaultdict(int)

    for record in records:

        class_counts[
            record["class_name"]
        ] += 1

    print()
    print("CLASS COUNTS")
    print("-" * 40)

    for class_name in [
        "drug",
        "handgun",
        "pistol",
        "knife"
    ]:

        print(
            f"{class_name:10s}: "
            f"{class_counts[class_name]}"
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, valid, test = split_images(
        records
    )

    print()
    print("FINAL SPLIT")
    print("-" * 40)

    print(
        f"Train : {len(train)}"
    )

    print(
        f"Valid : {len(valid)}"
    )

    print(
        f"Test  : {len(test)}"
    )

    print(
        f"Total : "
        f"{len(train) + len(valid) + len(test)}"
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    create_directories()

    write_data_yaml()

    manifest = []

    write_split(
        train,
        "train",
        manifest
    )

    write_split(
        valid,
        "valid",
        manifest
    )

    write_split(
        test,
        "test",
        manifest
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_ROOT
        / "split_manifest.csv"
    )

    fieldnames = [
        "split",
        "class",
        "class_id",
        "source_image",
        "output_image",
        "output_label",
        "annotation_type",
        "object_count",
    ]

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            manifest
        )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_ROOT
        / "split_summary.csv"
    )

    summary_rows = []

    for split_name, split_records in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:

        for class_name in [
            "drug",
            "handgun",
            "pistol",
            "knife"
        ]:

            count = sum(
                1
                for r in split_records
                if r["class_name"] == class_name
            )

            objects = sum(
                len(r["annotations"])
                for r in split_records
                if r["class_name"] == class_name
            )

            summary_rows.append({
                "split": split_name,
                "class": class_name,
                "class_id": CLASS_MAP[
                    class_name
                ],
                "images": count,
                "objects": objects,
            })

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class",
                "class_id",
                "images",
                "objects",
            ]
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMBINED DATASET CREATED")
    print("=" * 70)

    print()
    print(
        f"Location:\n{OUTPUT_ROOT}"
    )

    print()
    print("Structure:")

    print(
        """
combined_dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── data.yaml
├── split_manifest.csv
└── split_summary.csv
"""
    )

    print(
        "Ready for YOLO training."
    )


if __name__ == "__main__":
    main()