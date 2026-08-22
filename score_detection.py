"""Score the pipeline's detector against AAU RainSnow's COCO ground truth.

Evaluates the exact detector configuration the demos run (YOLOv8n, default
confidence, vehicle+person classes) on the 2,198 annotated rain/snow/night
frames, using greedy IoU matching at the standard 0.5 threshold. Reports
precision / recall / F1 for two groups:

* vehicles: GT {car, bus, truck, motorbike} vs predictions {2, 3, 5, 7}
* persons:  GT {person}                     vs prediction  {0}

GT bicycles are excluded — the signal pipeline's class filter deliberately
does not count bicycles as motor-vehicle demand.

Usage:
    python score_detection.py [--limit N] [--json-out scores.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GT_JSON = ROOT / "datasets" / "auurainsnow" / "aauRainSnow-rgb.json"
FRAMES_ROOT = ROOT / "datasets" / "auurainsnow"

VEHICLE_GT = {"car", "bus", "truck", "motorbike"}
PERSON_GT = {"person"}


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_boxes, pred_boxes, threshold=0.5):
    """Return (true_positives); each GT matches at most one prediction."""

    matched_gt, matched_pred = set(), set()
    pairs = sorted(
        ((iou(g, p), gi, pi)
         for gi, g in enumerate(gt_boxes)
         for pi, p in enumerate(pred_boxes)),
        reverse=True,
    )
    tp = 0
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp += 1
    return tp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="score only the first N images")
    parser.add_argument("--json-out", help="write results as JSON")
    args = parser.parse_args(argv)

    import cv2
    from smart_traffic_system import DetectorConfig, VehicleDetector

    data = json.loads(GT_JSON.read_text())
    categories = {c["id"]: c["name"] for c in data["categories"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    detector = VehicleDetector(DetectorConfig(classes=[0, 2, 3, 5, 7]))

    images = data["images"][: args.limit] if args.limit else data["images"]
    # totals[group][location] = [tp, n_gt, n_pred]
    totals = {"vehicles": defaultdict(lambda: [0, 0, 0]),
              "persons": defaultdict(lambda: [0, 0, 0])}
    skipped = 0
    started = time.time()

    for index, image_info in enumerate(images):
        path = FRAMES_ROOT / image_info["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            skipped += 1
            continue
        location = image_info["file_name"].split("/")[0]

        gt_vehicles, gt_persons = [], []
        for ann in annotations[image_info["id"]]:
            name = categories[ann["category_id"]]
            if name in VEHICLE_GT:
                gt_vehicles.append(ann["bbox"])
            elif name in PERSON_GT:
                gt_persons.append(ann["bbox"])

        detections = detector.detect_vehicles(frame)
        pred_vehicles = [d.bbox for d in detections if d.class_id != 0]
        pred_persons = [d.bbox for d in detections if d.class_id == 0]

        for group, gt, pred in (("vehicles", gt_vehicles, pred_vehicles),
                                ("persons", gt_persons, pred_persons)):
            tp = greedy_match(gt, pred)
            bucket = totals[group][location]
            bucket[0] += tp
            bucket[1] += len(gt)
            bucket[2] += len(pred)

        if (index + 1) % 250 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{len(images)} frames "
                  f"({rate:.1f}/s, ~{(len(images) - index - 1) / rate:.0f}s left)",
                  flush=True)

    def metrics(tp, n_gt, n_pred):
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    results = {"images_scored": len(images) - skipped, "skipped": skipped,
               "iou_threshold": 0.5, "groups": {}}
    print(f"\nScored {len(images) - skipped} frames (skipped {skipped})")
    for group, buckets in totals.items():
        tp = sum(b[0] for b in buckets.values())
        n_gt = sum(b[1] for b in buckets.values())
        n_pred = sum(b[2] for b in buckets.values())
        p, r, f1 = metrics(tp, n_gt, n_pred)
        print(f"\n{group.upper()}  (GT objects: {n_gt}, predictions: {n_pred})")
        print(f"  overall: precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")
        results["groups"][group] = {
            "gt": n_gt, "pred": n_pred, "tp": tp,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "per_location": {},
        }
        for location in sorted(buckets):
            lp, lr, lf1 = metrics(*buckets[location])
            print(f"    {location:<14} P {lp:.3f}  R {lr:.3f}  F1 {lf1:.3f} "
                  f"(GT {buckets[location][1]})")
            results["groups"][group]["per_location"][location] = {
                "precision": round(lp, 4), "recall": round(lr, 4),
                "f1": round(lf1, 4), "gt": buckets[location][1],
            }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nresults written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
