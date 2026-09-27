"""
BTPS_V1 - Complete EDA / Annotation Discovery

Run from BTPS_V1/preprocessing:
    python3 02_complete_eda_FIXED.py

Designed for the current structure, including:
    data/Drug Detection/train/...
    data/Knife/Knife_Dataset/images + labels
    data/knives/train/...
    data/pistol/export/_annotations.coco.json
    data/handgun/...

It supports YOLO TXT labels and COCO JSON annotations.
It NEVER modifies the original datasets.
"""

from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import math
import re
import csv
import sys

try:
    from PIL import Image
except ImportError:
    print("Install Pillow first: python3 -m pip install pillow")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUT = DATA_ROOT / "eda"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IGNORE = {"eda", "__pycache__", ".git"}


def rel(p):
    try:
        return str(p.relative_to(DATA_ROOT))
    except ValueError:
        return str(p)


def stem(p):
    return p.stem.lower()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def images_under(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
        and not any(x.lower() in IGNORE for x in p.parts)
    )


def yaml_names(yaml):
    """Small parser for common Roboflow/Ultralytics data.yaml files."""
    out = {}
    try:
        text = yaml.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out

    m = re.search(r"(?ms)^\s*names\s*:\s*\[(.*?)\]", text)
    if m:
        for i, v in enumerate(m.group(1).split(",")):
            v = v.strip().strip("'\"")
            if v:
                out[i] = v
        return out

    inside = False
    for line in text.splitlines():
        if re.match(r"^\s*names\s*:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if m:
                out[int(m.group(1))] = m.group(2).strip().strip("'\"")
            elif line.strip() and not line.startswith((" ", "\t")):
                inside = False
    return out


def parse_yolo(path):
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return rows
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        a = line.split()
        if len(a) != 5:
            rows.append({"valid": False, "line": n, "reason": "expected_5_values", "raw": line})
            continue
        try:
            c, x, y, w, h = int(float(a[0])), *map(float, a[1:])
        except Exception:
            rows.append({"valid": False, "line": n, "reason": "non_numeric", "raw": line})
            continue
        ok = 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        rows.append({
            "valid": ok, "line": n,
            "reason": "" if ok else "coordinates_outside_0_1",
            "class_id": c, "x": x, "y": y, "w": w, "h": h,
            "area": w*h, "aspect": w/h if h else math.inf
        })
    return rows


def parse_coco(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        return None
    cats = {int(x["id"]): str(x.get("name", x["id"])) for x in data.get("categories", []) if "id" in x}
    image_meta = {x["id"]: x for x in data.get("images", []) if "id" in x and "file_name" in x}
    by_stem = defaultdict(list)
    for ann in data.get("annotations", []):
        iid = ann.get("image_id")
        if iid in image_meta:
            by_stem[Path(str(image_meta[iid]["file_name"])).stem.lower()].append(ann)
    return {"path": path, "cats": cats, "images": image_meta, "anns": dict(by_stem)}


print("=" * 78)
print("BTPS_V1 COMPLETE EDA")
print("=" * 78)
print("Data root:", DATA_ROOT)
print("Output   :", OUT)

if not DATA_ROOT.exists():
    raise SystemExit(f"Data directory not found: {DATA_ROOT}")

# ------------------------------------------------------------
# 1. Discover annotation sources everywhere under data/
# ------------------------------------------------------------
yolo_files = [
    p for p in DATA_ROOT.rglob("*.txt")
    if p.is_file() and not p.name.lower().startswith("readme")
]

coco = []
for p in DATA_ROOT.rglob("*.json"):
    if p.is_file():
        x = parse_coco(p)
        if x:
            coco.append(x)

# index YOLO labels by exact filename stem
by_yolo_stem = defaultdict(list)
for p in yolo_files:
    by_yolo_stem[stem(p)].append(p)

print(f"YOLO txt files found : {len(yolo_files)}")
print(f"COCO json files found: {len(coco)}")

# ------------------------------------------------------------
# 2. Dataset roots = first-level directories under data/
#    (this is safer for your current layout than treating every
#    train/images folder as a separate dataset)
# ------------------------------------------------------------
roots = [
    p for p in DATA_ROOT.iterdir()
    if p.is_dir() and p.name.lower() not in IGNORE
]

# Do not treat the EDA output as a dataset.
roots = [p for p in roots if p.name.lower() != "eda"]

# ------------------------------------------------------------
# 3. Find the best annotation for an image.
#    Score favors annotations physically inside the same dataset root.
# ------------------------------------------------------------
def same_root_score(image, annotation):
    try:
        ir = image.relative_to(DATA_ROOT).parts
        ar = annotation.relative_to(DATA_ROOT).parts
    except ValueError:
        return 0
    score = 0
    for a, b in zip(ir, ar):
        if a.lower() == b.lower():
            score += 10
        else:
            break
    return score


def yolo_candidates(image):
    return by_yolo_stem.get(stem(image), [])


def best_yolo(image, root):
    candidates = yolo_candidates(image)
    if not candidates:
        return None
    scored = []
    for p in candidates:
        s = same_root_score(image, p)
        # Strong bonus for conventional images/labels pairing.
        if image.parent.name.lower() == "images" and p.parent.name.lower() == "labels":
            if p.parent.parent == image.parent.parent:
                s += 1000
        if p.parent == image.parent:
            s += 1200
        scored.append((s, p))
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return scored[0][1]


def best_coco(image, root):
    matches = []
    for c in coco:
        if stem(image) in c["anns"]:
            p = c["path"]
            s = same_root_score(image, p)
            if str(p).startswith(str(root)):
                s += 1000
            matches.append((s, c))
    matches.sort(key=lambda x: (-x[0], str(x[1]["path"])))
    return matches[0][1] if matches else None

# ------------------------------------------------------------
# Output structures
# ------------------------------------------------------------
summary = []
mapping_summary = []
image_map = []
class_counts = Counter()
bbox_rows = []
dim_counts = Counter()
invalid_rows = []
hashes = defaultdict(list)

# ------------------------------------------------------------
# 4. Process every dataset root
# ------------------------------------------------------------
for root in roots:
    imgs = images_under(root)
    if not imgs:
        continue

    dataset = rel(root)
    names = {}
    for y in root.rglob("*.yaml"):
        names.update(yaml_names(y))
    for y in root.rglob("*.yml"):
        names.update(yaml_names(y))

    annotated = 0
    missing = 0
    empty = 0
    invalid = 0
    yolo_n = 0
    coco_n = 0
    objects = 0
    obj_per_image = []
    local_hashes = defaultdict(list)

    source_counts = Counter()

    print("\n" + "-" * 78)
    print("DATASET:", dataset)
    print("Images :", len(imgs))

    for image in imgs:
        # image validation
        try:
            with Image.open(image) as im:
                w_img, h_img = im.size
                im.verify()
        except Exception as e:
            invalid_rows.append({
                "dataset": dataset, "image": rel(image),
                "problem": "corrupt_image", "details": str(e)
            })
            continue

        dim_counts[(dataset, w_img, h_img)] += 1
        digest = md5(image)
        local_hashes[digest].append(image)
        hashes[digest].append(image)

        # Prefer YOLO if it can be matched; otherwise try COCO.
        yolo = best_yolo(image, root)
        csrc = best_coco(image, root)
        anns = []
        source = ""
        source_file = ""

        if yolo:
            source = "YOLO"
            source_file = rel(yolo)
            yolo_n += 1
            recs = parse_yolo(yolo)
            if not recs:
                empty += 1
            for r in recs:
                if not r.get("valid"):
                    invalid += 1
                    invalid_rows.append({
                        "dataset": dataset, "image": rel(image),
                        "problem": "invalid_yolo_annotation",
                        "details": f"{rel(yolo)} line {r.get('line')}: {r.get('reason')}"
                    })
                    continue
                cid = r["class_id"]
                cname = names.get(cid, f"class_{cid}")
                anns.append((cid, cname, r["x"], r["y"], r["w"], r["h"], r["area"], r["aspect"]))
        elif csrc:
            source = "COCO"
            source_file = rel(csrc["path"])
            coco_n += 1
            for ann in csrc["anns"].get(stem(image), []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    invalid += 1
                    invalid_rows.append({
                        "dataset": dataset, "image": rel(image),
                        "problem": "invalid_coco_bbox", "details": str(bbox)
                    })
                    continue
                x0, y0, bw, bh = map(float, bbox)
                if bw <= 0 or bh <= 0 or w_img <= 0 or h_img <= 0:
                    invalid += 1
                    continue
                wn, hn = bw / w_img, bh / h_img
                xn, yn = (x0 + bw/2) / w_img, (y0 + bh/2) / h_img
                cid = ann.get("category_id")
                cname = csrc["cats"].get(int(cid), f"class_{cid}")
                anns.append((cid, cname, xn, yn, wn, hn, wn*hn, wn/hn if hn else math.inf))
        else:
            missing += 1

        if anns:
            annotated += 1
        nobj = len(anns)
        objects += nobj
        obj_per_image.append(nobj)
        source_counts[source or "MISSING"] += 1

        image_map.append({
            "dataset": dataset,
            "image": rel(image),
            "annotation_type": source or "NONE",
            "annotation_file": source_file,
            "status": "ANNOTATED" if anns else ("EMPTY_LABEL" if source else "MISSING_ANNOTATION"),
            "objects": nobj,
            "image_width": w_img,
            "image_height": h_img
        })

        for cid, cname, xn, yn, wn, hn, area, aspect in anns:
            class_counts[(dataset, str(cid), cname)] += 1
            bbox_rows.append({
                "dataset": dataset, "image": rel(image),
                "annotation_type": source, "annotation_file": source_file,
                "class_id": cid, "class_name": cname,
                "x_center_norm": xn, "y_center_norm": yn,
                "width_norm": wn, "height_norm": hn,
                "area_norm": area, "bbox_aspect_ratio": aspect,
                "image_width": w_img, "image_height": h_img
            })

    dup_groups = sum(1 for x in local_hashes.values() if len(x) > 1)
    avg_obj = objects / len(imgs) if imgs else 0
    coverage = annotated / len(imgs) * 100 if imgs else 0

    print("YOLO annotated:", yolo_n)
    print("COCO annotated:", coco_n)
    print("Missing:", missing)
    print("Empty labels:", empty)
    print("Invalid annotations:", invalid)
    print("Objects:", objects)
    print("Avg objects/image:", round(avg_obj, 3))
    print("Duplicate groups:", dup_groups)
    print("Coverage:", round(coverage, 2), "%")

    summary.append({
        "dataset": dataset,
        "images": len(imgs),
        "annotated_images": annotated,
        "missing_annotations": missing,
        "empty_labels": empty,
        "invalid_annotations": invalid,
        "yolo_annotated_images": yolo_n,
        "coco_annotated_images": coco_n,
        "total_objects": objects,
        "average_objects_per_image": round(avg_obj, 4),
        "duplicate_groups": dup_groups,
        "annotation_coverage_percent": round(coverage, 2)
    })

    for src, count in source_counts.items():
        mapping_summary.append({
            "dataset": dataset,
            "annotation_source_type": src,
            "matched_images": count,
            "percentage": round(count / len(imgs) * 100, 2),
            "status": "STRONG_MATCH" if count / len(imgs) >= .9 else ("PARTIAL_MATCH" if count / len(imgs) >= .5 else "WEAK_MATCH")
        })

# ------------------------------------------------------------
# 5. Duplicate / leakage report
# ------------------------------------------------------------
leakage = []
for digest, paths in hashes.items():
    if len(paths) < 2:
        continue
    datasets = sorted({p.relative_to(DATA_ROOT).parts[0] for p in paths})
    leakage.append({
        "md5": digest,
        "copies": len(paths),
        "datasets": " | ".join(datasets),
        "cross_dataset_duplicate": "YES" if len(datasets) > 1 else "NO",
        "paths": " | ".join(rel(p) for p in paths)
    })

# ------------------------------------------------------------
# 6. CSV writer
# ------------------------------------------------------------
def write_csv(name, rows, fields=None):
    path = OUT / name
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["message"]
    if not rows:
        rows = [{fields[0]: "No records found"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Created:", path)

write_csv("dataset_summary.csv", summary)
write_csv("label_mapping.csv", mapping_summary)
write_csv("image_annotation_mapping.csv", image_map)
write_csv("class_distribution.csv", [
    {"dataset": d, "class_id": c, "class_name": n, "object_count": k}
    for (d, c, n), k in sorted(class_counts.items())
])
write_csv("bbox_statistics.csv", bbox_rows)
write_csv("image_dimensions.csv", [
    {"dataset": d, "width": w, "height": h, "count": n,
     "aspect_ratio": round(w/h, 4) if h else ""}
    for (d, w, h), n in sorted(dim_counts.items())
])
write_csv("duplicates_and_leakage.csv", leakage)
write_csv("invalid_annotations.csv", invalid_rows)

# ------------------------------------------------------------
# 7. TXT guide report
# ------------------------------------------------------------
total_images = sum(x["images"] for x in summary)
total_ann = sum(x["annotated_images"] for x in summary)
total_missing = sum(x["missing_annotations"] for x in summary)
total_objects = sum(x["total_objects"] for x in summary)

lines = [
    "BTPS_V1 COMPLETE EDA REPORT",
    "=" * 78,
    "",
    f"Data root: {DATA_ROOT}",
    f"Datasets analyzed: {len(summary)}",
    f"Total images: {total_images}",
    f"Annotated images: {total_ann}",
    f"Images without matched annotations: {total_missing}",
    f"Total valid annotated objects: {total_objects}",
    "",
    "DATASET SUMMARY",
    "-" * 78,
]
for x in summary:
    lines += [
        f"{x['dataset']}",
        f"  images={x['images']}",
        f"  annotated_images={x['annotated_images']}",
        f"  missing_annotations={x['missing_annotations']}",
        f"  empty_labels={x['empty_labels']}",
        f"  invalid_annotations={x['invalid_annotations']}",
        f"  YOLO_images={x['yolo_annotated_images']}",
        f"  COCO_images={x['coco_annotated_images']}",
        f"  objects={x['total_objects']}",
        f"  avg_objects_per_image={x['average_objects_per_image']}",
        f"  duplicate_groups={x['duplicate_groups']}",
        f"  annotation_coverage={x['annotation_coverage_percent']}%",
        "",
    ]

lines += ["ANNOTATION SOURCE MATCHING", "-" * 78]
for x in mapping_summary:
    lines.append(
        f"{x['dataset']} -> {x['annotation_source_type']} | "
        f"{x['matched_images']} images | {x['percentage']}% | {x['status']}"
    )

lines += ["", "CLASS DISTRIBUTION", "-" * 78]
if class_counts:
    for (d, c, n), k in sorted(class_counts.items()):
        lines.append(f"{d} | class {c} | {n} | {k} objects")
else:
    lines.append("No valid annotations found.")

lines += ["", "DUPLICATES / LEAKAGE", "-" * 78]
if leakage:
    for x in leakage:
        lines.append(
            f"{x['cross_dataset_duplicate']} | {x['copies']} copies | {x['datasets']} | {x['paths']}"
        )
else:
    lines.append("No duplicate images found by MD5 hash.")

lines += [
    "",
    "IMPORTANT INTERPRETATION",
    "-" * 78,
    "The previous EDA reported labels as missing because it only looked for a narrow label layout.",
    "This version searches recursively and supports both YOLO TXT labels and COCO JSON annotations.",
    "For the pistol folder shown in the screenshot, _annotations.coco.json is therefore treated as a",
    "possible annotation source instead of declaring every pistol image unlabeled.",
    "",
    "Do not merge or train until label_mapping.csv and image_annotation_mapping.csv are checked.",
    "Do not delete any original files based only on this report.",
    "",
    "Recommended next step:",
    "1. Run this script.",
    "2. Open data/eda/label_mapping.csv.",
    "3. Open data/eda/dataset_summary.csv.",
    "4. Check data/eda/class_distribution.csv and bbox_statistics.csv.",
    "5. Then we can build the cleaning + unified training dataset.",
]

report = OUT / "EDA_REPORT.txt"
report.write_text("\n".join(lines), encoding="utf-8")
print("Created:", report)

print("\n" + "=" * 78)
print("EDA COMPLETE")
print("Open in VS Code:", OUT)
print("=" * 78)


# ============================================================
# VISUALIZATION + SPLIT PLAN OUTPUT
# ============================================================
# This section runs after the EDA CSVs are created.
# It creates PNG visualizations without modifying the datasets.
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("\nVisualization skipped.")
    print("Install dependencies with:")
    print("python3 -m pip install pandas matplotlib")
else:
    VIZ_DIR = OUT / "visualizations"
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    def save_bar(data, x, y, title, filename, xlabel=None, ylabel=None,
                 rotate=False, horizontal=False):
        if data.empty:
            return

        plt.figure(figsize=(10, 6))

        if horizontal:
            data = data.sort_values(y)
            plt.barh(data[x].astype(str), data[y])
        else:
            plt.bar(data[x].astype(str), data[y])
            if rotate:
                plt.xticks(rotation=45, ha="right")

        plt.title(title)
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / filename, dpi=200)
        plt.close()

    # 1. Images per dataset
    summary_path = OUT / "dataset_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)

        save_bar(
            summary,
            "dataset",
            "images",
            "Images per dataset",
            "01_images_per_dataset.png",
            ylabel="Number of images",
            rotate=True,
            horizontal=len(summary) >= 6
        )

        # 2. Annotation coverage
        save_bar(
            summary,
            "dataset",
            "annotation_coverage_percent",
            "Annotation coverage by dataset",
            "02_annotation_coverage.png",
            ylabel="Coverage (%)",
            rotate=True,
            horizontal=len(summary) >= 6
        )

        # 3. Objects per dataset
        save_bar(
            summary,
            "dataset",
            "total_objects",
            "Annotated objects per dataset",
            "03_objects_per_dataset.png",
            ylabel="Number of objects",
            rotate=True,
            horizontal=len(summary) >= 6
        )

    # 4. Class distribution
    class_path = OUT / "class_distribution.csv"
    if class_path.exists():
        classes = pd.read_csv(class_path)

        if not classes.empty and "class_name" in classes.columns:
            class_totals = (
                classes.groupby("class_name", as_index=False)["object_count"]
                .sum()
                .sort_values("object_count", ascending=False)
            )

            save_bar(
                class_totals,
                "class_name",
                "object_count",
                "Overall object class distribution",
                "04_class_distribution.png",
                ylabel="Number of annotated objects",
                horizontal=len(class_totals) >= 6
            )

    # 5. Image resolution distribution
    dim_path = OUT / "image_dimensions.csv"
    if dim_path.exists():
        dims = pd.read_csv(dim_path)

        if not dims.empty:
            dims["resolution"] = (
                dims["width"].astype(str) + "x" +
                dims["height"].astype(str)
            )

            top_dims = (
                dims.groupby("resolution", as_index=False)["count"]
                .sum()
                .sort_values("count", ascending=False)
                .head(15)
            )

            save_bar(
                top_dims,
                "resolution",
                "count",
                "Most common image resolutions",
                "05_image_resolutions.png",
                ylabel="Number of images",
                rotate=True
            )

    # 6. Bounding-box size distribution
    bbox_path = OUT / "bbox_statistics.csv"
    if bbox_path.exists():
        bbox = pd.read_csv(bbox_path)

        if not bbox.empty:
            bbox["area_norm"] = pd.to_numeric(
                bbox["area_norm"], errors="coerce"
            )

            bbox = bbox.dropna(subset=["area_norm"])

            if not bbox.empty:
                # Histogram
                plt.figure(figsize=(10, 6))
                plt.hist(bbox["area_norm"], bins=30)
                plt.title("Bounding-box normalized area distribution")
                plt.xlabel("Bounding-box area / image area")
                plt.ylabel("Number of objects")
                plt.tight_layout()
                plt.savefig(
                    VIZ_DIR / "06_bbox_area_distribution.png",
                    dpi=200
                )
                plt.close()

            # 7. Bounding-box width vs height
            if {"width_norm", "height_norm"}.issubset(bbox.columns):
                plt.figure(figsize=(8, 7))
                plt.scatter(
                    bbox["width_norm"],
                    bbox["height_norm"],
                    alpha=0.35,
                    s=12
                )
                plt.title("Bounding-box width vs height")
                plt.xlabel("Normalized box width")
                plt.ylabel("Normalized box height")
                plt.tight_layout()
                plt.savefig(
                    VIZ_DIR / "07_bbox_width_vs_height.png",
                    dpi=200
                )
                plt.close()

            # 8. Objects per image by dataset
            if "dataset" in bbox.columns:
                obj_per_image = (
                    bbox.groupby(
                        ["dataset", "image"],
                        as_index=False
                    )
                    .size()
                    .rename(columns={"size": "objects"})
                )

                obj_summary = (
                    obj_per_image.groupby(
                        "dataset", as_index=False
                    )["objects"]
                    .mean()
                    .sort_values("objects", ascending=False)
                )

                save_bar(
                    obj_summary,
                    "dataset",
                    "objects",
                    "Average objects per annotated image",
                    "08_average_objects_per_image.png",
                    ylabel="Average objects / image",
                    rotate=True,
                    horizontal=len(obj_summary) >= 6
                )

    # 9. Duplicate / leakage summary
    dup_path = OUT / "duplicates_and_leakage.csv"
    if dup_path.exists():
        dup = pd.read_csv(dup_path)

        if not dup.empty and "potential_cross_dataset_duplicate" in dup.columns:
            counts = (
                dup["potential_cross_dataset_duplicate"]
                .value_counts()
                .rename_axis("duplicate_type")
                .reset_index(name="count")
            )

            save_bar(
                counts,
                "duplicate_type",
                "count",
                "Duplicate image groups",
                "09_duplicate_groups.png",
                ylabel="Number of duplicate groups"
            )

    # ========================================================
    # PROPOSED COMBINED TRAINING SPLIT
    # ========================================================
    #
    # This is a PLAN, not a claim about the final number of images.
    # The final counts must be calculated AFTER:
    #   1. annotation verification
    #   2. invalid-image removal
    #   3. duplicate removal
    #   4. class remapping
    #
    # Recommended first BTPS experiment:
    #   80% train
    #   10% validation
    #   10% test
    #
    # Split at IMAGE level, never at bounding-box level.
    # Duplicate groups must stay in the same split.
    #
    split_plan = pd.DataFrame([
        {
            "split": "train",
            "percentage": 80,
            "purpose": "Model learning"
        },
        {
            "split": "valid",
            "percentage": 10,
            "purpose": "Hyperparameter/model selection"
        },
        {
            "split": "test",
            "percentage": 10,
            "purpose": "Final unseen evaluation"
        }
    ])

    split_plan.to_csv(
        OUT / "combined_dataset_split_plan.csv",
        index=False
    )

    # Human-readable plan.
    plan_text = """BTPS_V1 COMBINED DATASET SPLIT PLAN

TARGET TRAINING DATASET
-----------------------
For the first object-detection experiment, the intended core classes are:

0 = drug
1 = handgun
2 = pistol

The datasets are combined into ONE unified dataset after annotation
verification and class-ID remapping.

KNIFE DATA
----------
Knife data should remain outside the first 3-class training dataset if
the goal is to use it as an external/generalization or robustness test.
Do not mix it into training merely because it is available.

SPLIT
-----
Train      = 80%
Validation = 10%
Test       = 10%

IMPORTANT
---------
The percentages are applied AFTER cleaning and deduplication.

Split at IMAGE level, not bounding-box level.

If two identical images occur in different source datasets, they must
not be allowed to cross train/validation/test boundaries.

Do not augment validation or test images.

Do not use test results to tune the model.

The final split counts will be calculated only after the complete EDA
confirms:
- annotation coverage
- class mapping
- invalid annotations
- duplicate images
- cross-dataset leakage
- usable image count

FINAL STRUCTURE
---------------
combined_dataset/
    images/
        train/
        valid/
        test/

    labels/
        train/
        valid/
        test/

    data.yaml

data.yaml classes:
    0: drug
    1: handgun
    2: pistol
"""

    (OUT / "COMBINED_DATASET_SPLIT_PLAN.txt").write_text(
        plan_text,
        encoding="utf-8"
    )

    print("\nVISUALIZATIONS CREATED IN:")
    print(VIZ_DIR)

    print("\nSPLIT PLAN CREATED:")
    print(OUT / "combined_dataset_split_plan.csv")
    print(OUT / "COMBINED_DATASET_SPLIT_PLAN.txt")
