import os
import cv2
import numpy as np
import pandas as pd

# IMAGE FILE/FOLDER FUNCTIONS

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

def sortImagesByName(folder_path):
    """
    Sorts a list of image paths by their file names.
    """
    return sorted([f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTS)])

def loadImagesFromFolder(path):

    """
    Loads all images from a specified folder in int8 format.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            img = None
    
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be loaded: {path}")
    
    return img

def loadEmbeddingsFromFolder(csv_path):
    """
    Read the ResNet18 CSV: return (names, feats) with feats as (N, 512) float32.
    """
    df = pd.read_csv(csv_path, header=None)
    names = df[0].astype(str).str.strip().tolist()
    feats = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if feats.shape[1] != 512 or np.isnan(feats).any():
        raise ValueError(f"Unexpected embedding shape or NaNs: {feats.shape}")
    print(f"Loaded {len(names)} embeddings from {csv_path}")
    return names, feats


# FEATURE HANDLING FUNCTIONS

# Trial Function
def tiny(img):
    """
    Placeholder: 8x8 colour thumbnail. Only used to test the pipeline.
    """
    return cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()

# Center 7x7 patch for image matching 
def center_patch(img, patch_size=7):
    """  
    Extracts the 7x7 center patch from the given image.
    """
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image dimensions ({h}, {w}) are smaller than the patch size ({patch_size}).")
    
    r = patch_size // 2
    y, x = h // 2, w // 2
    patch = img[y - r : y + r + 1, x - r : x + r + 1]
    
    return patch.astype(np.float32).ravel()

# R-G Histogram
def rg_hist(img, bins=16):
    """
    Computes a 2D histogram of the R and G channels of the image.
    """
    img = img.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    s = r + g + b
    mask = s > 0

    # For a completely black image, return a zero histogram
    if not np.any(mask):
        return np.zeros((bins * bins), dtype=np.float32)  # Return a zero histogram if no valid pixels
    
    rn = (r[mask] / s[mask]).astype(np.float32).reshape(-1, 1)
    gn = (g[mask] / s[mask]).astype(np.float32).reshape(-1, 1)

    # Compute 2D histogram of normalized R and G channels
    hist = cv2.calcHist([rn, gn], [0, 1], None, [bins, bins], [0, 1.0001, 0, 1.0001])
    hist = hist.ravel().astype(np.float32)  # Normalize histogram
    hist /= np.sum(hist)  # Normalize to sum to 1

    return hist

# Multi-Histogram Helper 
def rgb_hist(region, bins=8):
    """
    Computes a 3D histogram of the R, G, and B channels of the image into a single feature vector.
    """
    q = (region // ( 256.0 // bins)).reshape(-1, 3).astype(np.int64)  # Normalize to [0, 1]
    pixel_idx = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]

    hist = np.bincount(pixel_idx, minlength=bins**3).astype(np.float32)
    total = np.sum(hist)

    if total > 0:
        hist = hist / total  # Normalize histogram to sum to 1

    return hist

# Multi-Histogram Feature for top and bottom halves of the image
def multi_hist(img):
    """Feature vector of 2 * HIST_LEN values: [top-half hist, bottom-half hist].

    The split uses this image's own height, so images of different sizes work.
    """
    h = img.shape[0]
    top = img[: h // 2]
    bottom = img[h // 2 :]
    return np.concatenate([rgb_hist(top), rgb_hist(bottom)])

# HSV Histogram Feature
def hsv_hist(img, bins=8):
    """
    Computes a 3D histogram of the H, S, and V channels of the image into a single feature vector.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    h = hsv_img[:, :, 0] * bins // 180  # Normalize H channel to [0, bins)
    s = hsv_img[:, :, 1] * bins // 256  # Normalize S channel to [0, bins)
    v = hsv_img[:, :, 2] * bins // 256  # Normalize V channel to [0, bins)
    q = (h * bins * bins + s * bins + v).reshape(-1, 1)

    hist = np.bincount(q[:, 0], minlength=bins**3).astype(np.float32)
    
    total = np.sum(hist)
    if total > 0:
        hist = hist / total  # Normalize histogram to sum to 1

    return hist

# Texture Feature
def sobel_hist(img, bins=16):
    """
    Computes a histogram of the Sobel gradient magnitudes of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx * sobelx + sobely * sobely).astype(np.float32)

    hist = np.histogram(magnitude, bins=bins, range=(0, 1443))[0].astype(np.float32)
    
    total = np.sum(hist)
    if total > 0:
        hist = hist / total  # Normalize histogram to sum to 1

    return hist

# Combining the hsv_hist and sobel_hist features into a single feature vector
def color_feature(img):
    """
    Combines HSV histogram and Sobel histogram features into a single feature vector.
    """
    hsv = hsv_hist(img)
    sobel = sobel_hist(img)
    return np.concatenate([hsv, sobel])


def center_weighted_hs(img, bins=16):
    """
    Computes a center-weighted histogram of the H and S channels of the image.
    The center region is given more weight in the histogram.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    h_bins = hsv_img[:, :, 0] * bins // 180  # Normalize H channel to [0, bins)
    s_bins = hsv_img[:, :, 1] * bins // 256  # Normalize S channel to [0, bins)

    # Create a weight mask that gives more weight to the center of the image
    H, W = img.shape[:2]
    y, x = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    distance_from_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
    weights = 1 - (distance_from_center / max_distance)  # Weight decreases with distance from center

    q = (h_bins * bins + s_bins).ravel()
    
    hist = np.bincount(q, weights=weights.ravel(), minlength=bins**2).astype(np.float32)
    
    total = np.sum(hist)
    if total > 0:
        hist = hist / total  # Normalize histogram to sum to 1

    return hist

def edge_orient_hist(img, bins=12):
    """
    Computes a histogram of edge orientations using the Sobel operator.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    orientation = (np.arctan2(sobely, sobelx) + np.pi) * (bins / (2 * np.pi))  # Normalize to [0, bins)
    
    hist = np.histogram(orientation.ravel(), bins=bins, range=(0, bins), weights=magnitude.ravel())[0].astype(np.float32)
    
    total = np.sum(hist)
    if total > 0:
        hist = hist / total  # Normalize histogram to sum to 1

    return hist

FEATURES = {"tiny": tiny, "baseline": center_patch, "rghist": rg_hist, "multihist": multi_hist, "colortext": color_feature}


# DISTANCE FUNCTIONS

# SSD (Sum of Squared Differences) distance between a query and database features
def ssd(q, db):
    """
    q: (D,), db: (N, D) -> (N,)
    """
    diff = db.astype(np.float32) - q.astype(np.float32)
    return np.sum(diff * diff, axis=1)

# Histogram intersection distance: 0 means identical. Used for RG histogram matching
def intersect(q, db):
    """Histogram intersection distance: 0 means identical."""
    return 1.0 - np.minimum(db, q).sum(axis=1)

# Multi-histogram distance: 0 means identical. Used in multi_hist feature matching
def multi_dist(q, db, w_top=0.5, w_bottom=0.5):
    """Distance between a query and every row of db, for multi_hist features.

    Each half is compared with histogram intersection (1 - sum of minimums),
    then the two distances are combined with the given weights. Returns one
    distance per database row; 0 means identical.
    """
    n = 512         # Number of bins per half of the histogram
    d_top = 1.0 - np.minimum(db[:, :n], q[:n]).sum(axis=1)
    d_bottom = 1.0 - np.minimum(db[:, n:], q[n:]).sum(axis=1)
    return w_top * d_top + w_bottom * d_bottom

# Color and texture distance: 0 means identical. Used in color_texture feature matching
def color_texture_dist(q, db, w_color=0.5, w_texture=0.5):
    """Distance between a query and every row of db, for color_texture features.

    Each half is compared with histogram intersection (1 - sum of minimums),
    then the two distances are combined with the given weights. Returns one
    distance per database row; 0 means identical.
    """
    n = 512         # Number of bins per half of the histogram
    d_color = 1.0 - np.minimum(db[:, :n], q[:n]).sum(axis=1)
    d_texture = 1.0 - np.minimum(db[:, n:], q[n:]).sum(axis=1)
    return w_color * d_color + w_texture * d_texture

def cosine_distance(q, db):
    """Computes the cosine distance between a query and every row of db.

    Cosine distance is defined as 1 - cosine similarity. Returns one
    distance per database row; 0 means identical.
    """
    q_norm = np.linalg.norm(q)
    db_norms = np.linalg.norm(db, axis=1)
    
    # Avoid division by zero
    if q_norm == 0:
        raise ValueError("Query vector has zero norm.")
    
    # Compute cosine similarity
    cosine_similarity = np.dot(db, q) / (db_norms * q_norm + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Convert to cosine distance
    return 1.0 - cosine_similarity


# =============================================================================
# CUSTOM DESIGN: SHOE RETRIEVER
# Feature row = [512 embedding | 256 centre-weighted colour | 12 edge orientation]
# Distance = weighted sum of three separate distances, each between 0 and 1.
# =============================================================================

EMB_LEN = 512
COLOR_LEN = 256
EDGE_LEN = 12


# --- Classic part of the shoe feature (computed from the image) --------------
def shoe_classic(img):
    """Centre-weighted H-S histogram plus edge-orientation histogram (268 values)."""
    return np.concatenate([center_weighted_hs(img), edge_orient_hist(img)])


# --- Distance: embedding + colour + edge, combined with weights --------------
def shoe_dist(q, db, w=(0.6, 0.25, 0.15)):
    """Weighted sum of three distances for rows laid out as
    [embedding | colour histogram | edge histogram]. 0 means identical.

    - embedding: cosine distance
    - colour:    1 - histogram intersection
    - edges:     half chi-square (between 0 and 1 for histograms that sum to 1)
    """
    a = EMB_LEN
    b = EMB_LEN + COLOR_LEN

    d_embed = cosine_distance(q[:a], db[:, :a])
    d_color = 1.0 - np.minimum(db[:, a:b], q[a:b]).sum(axis=1)
    d_edge = 0.5 * ((db[:, b:] - q[b:]) ** 2 / (db[:, b:] + q[b:] + 1e-9)).sum(axis=1)

    return w[0] * d_embed + w[1] * d_color + w[2] * d_edge

DISTANCES = {"ssd": ssd, "intersect": intersect, "multi": multi_dist, "colortext": color_texture_dist, "cosine": cosine_distance, "shoe": shoe_dist}


# DISPLAY FUNCTIONS

def rank(dists, names, target_name, n, worst=False):
    """
    Ranks the distances and returns the top n results.
    """

    order = np.argsort(dists, kind = "stable")
    if worst:
        order = order[::-1]  # Sort in descending order for worst matches
    
    out = []
    for i in order:
        if names[i] == target_name:
            continue
        out.append((names[i], float(dists[i])))
        if len(out) == n:
            break
    
    return out

def tile(image, label, h=180):
    """
    Creates a tile of the image with a label at the top. The image is resized to have height h.
    """
    scale = h / image.shape[0]
    t = cv2.resize(image, (max(1, int(image.shape[1] * scale)), h))
    cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t

def row(tiles):
    return cv2.hconcat(tiles)

def stack_rows(rows):
    w= max([r.shape[1] for r in rows])
    padded = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
    return cv2.vconcat(padded)