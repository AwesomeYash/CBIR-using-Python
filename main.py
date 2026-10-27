import argparse
import os
import cv2
import numpy as np
from utils import (FEATURES, DISTANCES, sortImagesByName, loadImagesFromFolder,
                   loadEmbeddingsFromFolder, rank, tile, row, stack_rows, shoe_classic, shoe_dist)

DEFAULT_DB = "olympus"

def load_or_none(path):
    """Your loader returns None on failure; this also catches an exception."""
    try:
        return loadImagesFromFolder(path)
    except Exception as e:
        print("error:", e)
        return None

def main():
    ap = argparse.ArgumentParser(description="Content-based image retrieval")
    ap.add_argument("--target", required=True, help="target image path or filename")
    ap.add_argument("--db", default=DEFAULT_DB, help="folder of database images")
    ap.add_argument("--feature", default="tiny", choices=sorted(list(FEATURES) + ["dnn", "shoe"]))
    ap.add_argument("--metric", default="ssd", choices=sorted(DISTANCES))
    ap.add_argument("--csv", default="ResNet18_olym.csv", help="ResNet Embeddings file")
    ap.add_argument("--n", type=int, default=3, help="number of matches")
    ap.add_argument("--worst", action="store_true", help="also show least similar")
    ap.add_argument("--save", help="write a montage PNG to this path")
    ap.add_argument("--show", action="store_true", help="open a window")
    args = ap.parse_args()

    # Resolve the target: use the path as given, else look inside the database folder
    target_path = args.target
    if not os.path.exists(target_path):
        target_path = os.path.join(args.db, args.target)
    target_name = os.path.basename(target_path)
    target_img = load_or_none(target_path)
    if target_img is None:
        raise SystemExit(f"Could not read target image: {target_path}")

    # --- Features: from the CSV (dnn) or computed from the images ---
    if args.feature == "dnn":
        names, feats = loadEmbeddingsFromFolder(args.csv)
        if target_name not in names:
            raise SystemExit(f"{target_name} not found in {args.csv}")
        q = feats[names.index(target_name)]
    
    elif args.feature == "shoe":
            names, emb = loadEmbeddingsFromFolder(args.csv)
            if target_name not in names:
                raise SystemExit(f"{target_name} not found in {args.csv}")

            # Loop in the CSV's order so the embedding and classic parts stay aligned.
            kept_names, rows_ = [], []
            for i, name in enumerate(names):
                img = load_or_none(os.path.join(args.db, name))
                if img is None:
                    print("skipped:", name)
                    continue                      # drops the embedding row too
                kept_names.append(name)
                rows_.append(np.concatenate([emb[i], shoe_classic(img)]))

            names = kept_names
            feats = np.stack(rows_)               # shape (N, 780)
            q = feats[names.index(target_name)]
    
    else:
        feature_fn = FEATURES[args.feature]
        q = feature_fn(target_img)

        names, feats = [], []
        for name in sortImagesByName(args.db):
            img = load_or_none(os.path.join(args.db, name))
            if img is None:
                print("skipped:", name)
                continue
            feats.append(feature_fn(img))
            names.append(name)
        if not names:
            raise SystemExit(f"No readable images found in: {args.db}")
        feats = np.stack(feats)

    # --- Distances ---
    dists = DISTANCES[args.metric](q, feats)

    # Rank and print
    top = rank(dists, names, target_name, args.n)
    print(f"Top {len(top)} matches for {target_name} ({args.feature}, {args.metric}):")
    for name, d in top:
        print(f"  {name}  {d:.4f}")

    bottom = []
    if args.worst:
        bottom = rank(dists, names, target_name, args.n, worst=True)
        print(f"Least similar {len(bottom)}:")
        for name, d in bottom:
            print(f"  {name}  {d:.4f}")

    # Montage and windows
    if args.save or args.show:
        def tiles_for(pairs):
            return [tile(load_or_none(os.path.join(args.db, nm)), f"{nm} {d:.3f}")
                    for nm, d in pairs]

        target_tile = tile(target_img, "TARGET " + target_name, h=360)
        top_row = row(tiles_for(top))
        bottom_row = row(tiles_for(bottom)) if bottom else None

        if args.save:
            rows = [row([tile(target_img, "TARGET " + target_name)] + tiles_for(top))]
            if bottom_row is not None:
                rows.append(bottom_row)
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, stack_rows(rows))

        if args.show:
            cv2.imshow("Target", target_tile)
            cv2.moveWindow("Target", 20, 20)
            cv2.imshow("Top matches", top_row)
            cv2.moveWindow("Top matches", 20 + target_tile.shape[1] + 20, 20)
            if bottom_row is not None:
                cv2.imshow("Least similar", bottom_row)
                cv2.moveWindow("Least similar", 20 + target_tile.shape[1] + 20, 300)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()