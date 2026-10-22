import argparse
import os

import cv2
import numpy as np

from utils import (FEATURES, DISTANCES, list_images, load_image,
                   rank, make_tile, make_row, stack_rows)

def main():
    ap = argparse.ArgumentParser(description="Content-based image retrieval")
    ap.add_argument("--target", required=True, help="target image path or filename")
    ap.add_argument("--db", required=True, help="folder of database images")
    ap.add_argument("--feature", default="tiny", choices=sorted(FEATURES))
    ap.add_argument("--metric", default="ssd", choices=sorted(DISTANCES))
    ap.add_argument("--n", type=int, default=3, help="number of matches")
    ap.add_argument("--worst", action="store_true", help="also show least similar")
    ap.add_argument("--save", help="write a montage PNG to this path")
    ap.add_argument("--show", action="store_true", help="open a window")
    args = ap.parse_args()

    # Resolve the target: use the path as given, else look in the database folder
    target_path = args.target
    if not os.path.exists(target_path):
        target_path = os.path.join(args.db, args.target)
    target_name = os.path.basename(target_path)
    target_img = load_image(target_path)

    feature_fn = FEATURES[args.feature]
    q = feature_fn(target_img)

    # Compute features for the whole database
    names, feats = [], []
    for name in list_images(args.db):
        try:
            feats.append(feature_fn(load_image(os.path.join(args.db, name))))
            names.append(name)
        except FileNotFoundError as e:
            print("skipped:", e)
    if not names:
        raise SystemExit("No images found in the database folder.")

    dists = DISTANCES[args.metric](q, np.stack(feats))

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

    # Montage
    if args.save or args.show:
        def tiles_for(pairs):
            return [make_tile(load_image(os.path.join(args.db, nm)), f"{nm} {d:.3f}")
                    for nm, d in pairs]

        sheet = make_row([make_tile(target_img, "TARGET " + target_name)] + tiles_for(top))
        rows = [make_row([make_tile(target_img, "TARGET " + target_name)] + tiles_for(top))]
        if bottom:
            rows.append(make_row(tiles_for(bottom)))
            sheet = stack_rows(rows)
        
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, sheet)
        if args.show:
            cv2.imshow("results", sheet)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()