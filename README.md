# Content-Based Image Retrieval (CBIR) in Python

A command-line image retrieval system. Given a target image and a folder of images, it finds the images most similar to the target using hand-built features (pixel patches, colour histograms) and distance metrics. Texture features, deep-network embeddings and a custom design are planned (see the roadmap).

This is a Python rebuild of a project I first wrote in C++, made to go back through the concepts.

**Status:** three feature types implemented, more in progress.

## How it works

For every query the program does four things:

1. Compute a feature vector for the target image.
2. Compute the same feature for every image in the database folder.
3. Compute the distance from the target to every database image.
4. Sort by distance (smaller means more similar), drop the target itself, and print the top N filenames.

Features and distances are plugged in by name, so adding a method means writing one function and registering it in a dictionary.

## What is implemented

| Feature (`--feature`) | Metric (`--metric`) | Description |
| --- | --- | --- |
| `baseline` | `ssd` | 7x7 pixel patch from the image centre (147 values), compared with sum of squared differences |
| `rghist` | `intersect` | 2D rg-chromaticity histogram (16x16 bins), compared with histogram intersection |
| `multihist` | `multi` | RGB histograms (8 bins per channel) of the top half and the bottom half, combined with equal weights |
| `tiny` | `ssd` | 8x8 thumbnail, used only to test the pipeline |

Notes on the methods:

- **SSD** is written in NumPy, not with an OpenCV distance function. Values are converted to `float32` before subtracting, so `uint8` never wraps around.
- **Histogram intersection** adds up the smaller value in each bin of two normalized histograms. The distance is `1 - intersection`.
- **rg-chromaticity** divides out brightness: r = R/(R+G+B) and g = G/(R+G+B). It only describes colour, so it cannot tell a blue sky from a blue sofa.
- **Multi-histogram** keeps a rough sense of layout (what is on top versus on the bottom) by comparing the two halves separately.

## Which distance for which feature

| Feature | Distance | Why |
| --- | --- | --- |
| Raw pixel values (`baseline`) | `ssd` | The vector is a list of pixel values compared position by position, and squaring penalizes large gaps. |
| Histogram (`rghist`) | `intersect` | A normalized histogram is a distribution, and intersection measures how much of it two images share, on a fixed scale from 0 to 1. |
| Several histograms (`multihist`) | `multi` | Intersection per region, then a weighted average, so each region counts separately and the weights are adjustable. |
| Deep embedding (planned) | cosine | An embedding's direction carries the meaning, and cosine ignores vector length. |

Rule of thumb: raw values go with SSD, histograms go with intersection, and embeddings go with cosine. A distance that does not fit the feature still runs without an error, but its results are misleading.

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

## Dataset

The image database is **not included** in this repository, and I do not redistribute it.

- Put the images in a folder named `olympus` next to `main.py`, or pass another folder with `--db`.
- The folder should contain `.jpg` images named like `pic.0164.jpg`.

## Usage

```powershell
python main.py --target pic.0164.jpg --feature rghist --metric intersect --n 3 --worst --show --save out\rghist_0164.png
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--target` | Target image: a path, or a filename inside the database folder | required |
| `--db` | Folder of database images | `olympus` |
| `--feature` | `tiny`, `baseline`, `rghist`, `multihist` | `tiny` |
| `--metric` | `ssd`, `intersect`, `multi` | `ssd` |
| `--n` | Number of matches to return | `3` |
| `--worst` | Also list the least similar images | off |
| `--save` | Write a montage PNG to this path | none |
| `--show` | Open windows for the target, the best matches and the least similar | off |

The program always prints the top N filenames with their distances.

### Example commands

```powershell
python main.py --target pic.1016.jpg --feature baseline  --metric ssd       --n 4 --worst --show --save out\baseline_1016.png
python main.py --target pic.0164.jpg --feature rghist    --metric intersect --n 3 --worst --show --save out\rghist_0164.png
python main.py --target pic.0274.jpg --feature multihist --metric multi     --n 3 --worst --show --save out\multihist_0274.png
```

## Verification

I have known answers for three queries, which I use as tests.

| Feature | Target | Expected top results | Status |
| --- | --- | --- | --- |
| `baseline` | `pic.1016` | `pic.0986`, `pic.0641`, `pic.0547`, `pic.1013` | to fill in |
| `rghist` | `pic.0164` | `pic.0080`, `pic.1032`, `pic.0461` | matches |
| `multihist` | `pic.0274` | `pic.0273`, `pic.1031`, `pic.0409` | to verify |

Other checks: every histogram sums to 1, a vector compared with itself has distance 0, and the target never appears in its own results.

## Project structure

```text
cbir-python/
  main.py            command-line entry point
  utils.py           image loading, features, distances, ranking, display
  requirements.txt
  README.md
  LICENSE
  .gitignore         excludes the environment, the images and the out/ folder
  out/               generated montages (not committed)
```

## What I learned so far

- A colour histogram describes what colours an image contains, not where they are or what objects are present. The rg histogram matched `pic.0461` (a red toy car on blue cloth) to a blue-sky target because both put similar amounts of colour in the same bins.
- Comparing `uint8` arrays needs care: subtraction wraps around, so convert to `float32` first.
- Normalizing histograms makes images of different sizes comparable.

## Roadmap

In order:

1. **Colour plus texture.** HSV colour histogram plus a histogram of Sobel gradient magnitudes, with equal weighting. Bin the magnitudes over their real range (0 to about 1,443), not 0 to 256.
2. **Deep network embeddings.** Read the 512-value ResNet18 vectors from a CSV and match with SSD and cosine distance.
3. **Deep versus classic features.** Compare the results on the same targets and ask whether the embedding is always better.
4. **A custom design for shoes.** Cosine distance on the embedding, a centre-weighted colour histogram and an edge-orientation histogram, combined with separate distances and explicit weights. Evaluate on held-out queries and show the least similar results.

## Potential additions

Engineering:

- Cache features to disk (`.npz`) so each feature is computed once, not on every query.
- Unit tests with `pytest`: self-distance is 0, histograms sum to 1, the target is excluded, and different image sizes do not crash.
- A `--patch-size` option for the centre patch, to compare patch sizes.
- Split `utils.py` into smaller modules if it grows past a few hundred lines.
- Evaluation helpers: precision at k and a table of methods against targets.

Retrieval extensions:

- More texture features: gradient-orientation histogram, a 2D orientation-by-magnitude histogram, co-occurrence (GLCM) features, Laws filters, Gabor filters, a Fourier power spectrum.
- Weight tuning for the multi-histogram and the custom design.
- A spatial colour layout (a grid of histograms) or spatial variance of colours.
- Depth Anything V2 depth maps to keep only the closer pixels in a histogram, focusing on the main object.
- A face-aware metric, other embeddings, or images that are not in the database.
- Approximate nearest-neighbour search and a timing comparison with brute force.
- A simple GUI for choosing a target and browsing results.

## License

MIT License. See `LICENSE`. The image dataset and the embeddings file are not covered by this license and are not included.
