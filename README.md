# Content-Based Image Retrieval (CBIR) in Python

A command-line image retrieval system. Given a target image and a folder of images, it finds the images most similar to the target. It supports hand-built features (pixel patches, colour and texture histograms), precomputed deep-network embeddings, and a custom retriever for shoe images that combines both kinds.

This is a Python rebuild of a project I first wrote in C++, made to go back through the concepts.

**Status:** all planned features are implemented.

## Project structure

```text
cbir-python/
  main.py            command-line entry point
  utils.py           image loading, features, distances, ranking, display
  requirements.txt
  README.md
  LICENSE
  .gitignore         excludes the environment, the data and the out/ folder
  out/               generated montages (not committed)
```

## How it works

For every query the program does four things:

1. Get a feature vector for the target image.
2. Get the same feature for every image in the database.
3. Compute the distance from the target to every database image.
4. Sort by distance (smaller means more similar), drop the target itself, and print the top N filenames.

Features and distances are plugged in by name, so adding a method means writing one function and registering it in a dictionary.

## Features

| `--feature` | What it is |
| --- | --- |
| `baseline` | 7x7 pixel patch from the image centre (147 values) |
| `rghist` | 2D rg-chromaticity histogram, 16x16 bins (256 values) |
| `multihist` | RGB histograms (8 bins per channel) of the top half and the bottom half (1,024 values) |
| `colortext` | HSV colour histogram (8x8x8 bins) plus a histogram of Sobel gradient magnitudes (16 bins), 528 values |
| `dnn` | Precomputed 512-value ResNet18 embedding, read from a CSV file |
| `shoe` | Embedding (512) plus a centre-weighted hue-saturation histogram (256) plus an edge-orientation histogram (12), 780 values |
| `tiny` | 8x8 thumbnail, used only to test the pipeline |

Notes on the methods:

- **SSD** is written in NumPy, not with an OpenCV distance function. Values are converted to `float32` before subtracting, so `uint8` never wraps around.
- **rg-chromaticity** divides out brightness: r = R/(R+G+B) and g = G/(R+G+B). It describes colour only, so it cannot tell a blue sky from a blue sofa.
- **Multi-histogram** keeps a rough sense of layout (what is on top versus on the bottom) by comparing the two halves separately.
- **Texture** comes from Sobel gradient magnitudes. They reach about 1,443 for 8-bit input, so the histogram range must cover that, not 0 to 255.
- **The shoe retriever** weights the centre of the image more, so the shoe counts more than the background. The edge-orientation histogram captures the elongated outline of a shoe.

## Which distance for which feature

| Feature | `--metric` | Why |
| --- | --- | --- |
| `baseline` | `ssd` | Raw pixel values are compared position by position, and squaring penalizes large gaps. |
| `rghist` | `intersect` | A normalized histogram is a distribution, and intersection measures how much of it two images share, on a fixed scale from 0 to 1. |
| `multihist` | `multidist` | Intersection per half, then a weighted average, so each region counts separately. |
| `colortext` | `colortext` | Intersection per part (colour, texture), then an equal-weight average, so neither dominates. |
| `dnn` | `cosine` | An embedding's direction carries the meaning, and cosine ignores vector length. `ssd` also works. |
| `shoe` | `shoe` | A separate distance per part (cosine, intersection, chi-square), each between 0 and 1, then a weighted sum. |

Rule of thumb: raw values go with SSD, histograms go with intersection, and embeddings go with cosine. When a feature mixes parts with different scales, compute a distance per part and combine them with weights, never one SSD on the whole vector. A mismatched pair still runs without an error, but its results are misleading.

## Requirements

- Python 3.13 (3.11 and 3.12 should also work)
- `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scipy`, `pytest`

Exact versions are in `requirements.txt`.

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv CBIR
.\CBIR\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first.

## Data

The image database and the embeddings file are **not included** in this repository, and I do not redistribute them.

- Put the images in a folder named `olympus` next to `main.py`, or pass another folder with `--db`. The images are `.jpg` files named like `pic.0164.jpg`.
- Put the embeddings in `ResNet18_olym.csv` next to `main.py`, or pass another file with `--csv`. Each row is a filename followed by 512 values, with no header.

## Usage

```powershell
python main.py --target pic.0164.jpg --feature rghist --metric intersect --n 3 --worst --show --save out\rghist_0164.png
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--target` | Target image: a path, or a filename inside the database folder | required |
| `--db` | Folder of database images | `olympus` |
| `--feature` | `tiny`, `baseline`, `rghist`, `multihist`, `colortext`, `dnn`, `shoe` | `tiny` |
| `--metric` | `ssd`, `intersect`, `multidist`, `colortext`, `cosine`, `shoe` | `ssd` |
| `--csv` | ResNet18 embeddings file (used by `dnn` and `shoe`) | `ResNet18_olym.csv` |
| `--n` | Number of matches to return | `3` |
| `--worst` | Also list the least similar images | off |
| `--save` | Write a montage PNG to this path | none |
| `--show` | Open windows for the target, the best matches and the least similar | off |

The program always prints the top N filenames with their distances.

### Example commands

```powershell
python main.py --target pic.1016.jpg --feature baseline  --metric ssd       --n 4 --worst --show --save out\baseline_1016.png
python main.py --target pic.0164.jpg --feature rghist    --metric intersect --n 3 --worst --show --save out\rghist_0164.png
python main.py --target pic.0274.jpg --feature multihist --metric multidist --n 3 --worst --show --save out\multihist_0274.png
python main.py --target pic.0535.jpg --feature colortext --metric colortext --n 3 --worst --show --save out\colortext_0535.png
python main.py --target pic.0893.jpg --feature dnn       --metric cosine    --n 3 --worst --show --save out\dnn_0893.png
python main.py --target pic.0758.jpg --feature shoe      --metric shoe      --n 5 --worst --show --save out\shoe_0758.png
```

## What I learned

- A colour histogram describes what colours an image contains, not where they are or what objects are present. The rg histogram matched `pic.0461` (a red toy car on blue cloth) to a blue-sky target because both put similar amounts of colour in the same bins.
- Embeddings match content (the same kind of object), while histograms match colour. Neither is always better.
- Comparing `uint8` arrays needs care: subtraction wraps around, so convert to `float32` first. The same goes for HSV values that get multiplied before binning, which need `int32`.
- A histogram range must cover the data. Values outside it are silently dropped, which happened with Sobel magnitudes over 0 to 255.
- Normalizing histograms makes images of different sizes comparable.
- Features on different scales should not share one distance. A separate distance per feature, each between 0 and 1, combined with explicit weights, keeps one feature from drowning out the others.

## Potential additions

Engineering:

- Cache features to disk (`.npz`) so each feature is computed once, not on every query. The classic features load about 1,100 images per run.
- Unit tests with `pytest`: self-distance is 0, histograms sum to 1, the target is excluded, and different image sizes do not crash.
- A `--patch-size` option for the centre patch, to compare patch sizes.
- Split `utils.py` into smaller modules, and rename `loadImagesFromFolder`, which loads a single image.
- Evaluation helpers: precision at k and a table of methods against targets.

Retrieval extensions:

- More texture features: a 2D orientation-by-magnitude histogram, co-occurrence (GLCM) features, Laws filters, Gabor filters, a Fourier power spectrum.
- Weight tuning for the multi-histogram, colour-plus-texture and shoe retrievers, evaluated on held-out queries.
- A spatial colour layout (a grid of histograms) or spatial variance of colours.
- Depth Anything V2 depth maps to keep only the closer pixels in a histogram, focusing on the main object.
- Other custom retrievers (for example faces or trash cans), or other embeddings.
- Approximate nearest-neighbour search and a timing comparison with brute force.
- A simple GUI for choosing a target and browsing results.

## License

MIT License. See `LICENSE`. The image dataset and the embeddings file are not covered by this license and are not included.
