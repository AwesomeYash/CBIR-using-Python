import os
import cv2
import numpy as np

# IMAGE FILE/FOLDER FUNCTIONS

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

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

# FEATURE HANDLING FUNCTIONS

def tiny(img):
    """
    Placeholder: 8x8 colour thumbnail. Only used to test the pipeline.
    """
    return cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()

# Center 7x7 patch for image matching 
def center_patch(img, patch_size=7):
    """  
    Extracts the 7x7 center patch from the given image. and compares it with other using SSD distance metric.
    """
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image dimensions ({h}, {w}) are smaller than the patch size ({patch_size}).")
    
    r = patch_size // 2
    y, x = h // 2, w // 2
    patch = img[y - r : y + r + 1, x - r : x + r + 1]
    
    return patch.astype(np.float32).ravel()

FEATURES = {"tiny": tiny, "baseline": center_patch}

def extract(name, img):
    return FEATURES[name](img)

# DISTANCE FUNCTIONS

def ssd(q, db):
    """
    q: (D,), db: (N, D) -> (N,)
    """
    diff = db.astype(np.float32) - q.astype(np.float32)
    return np.sum(diff * diff, axis=1)


DISTANCES = {"ssd": ssd}

def rank(dists, names, target_name, n, worst=False):
    """
    Ranks the distances and returns the top n results.

    Args:
        dists (np.ndarray): Array of distances.
        names (list): List of corresponding names.
        target_name (str): The name to compare against.
        n (int): Number of top results to return.
        worst (bool): If True, returns the worst matches instead of the best.
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

# Display Functions 
def tile(image, label, h=180):
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
