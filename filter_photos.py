# filter_photos.py (Enhanced with Baseline Sampling)

import os
import argparse
import cv2
import shutil
import numpy as np
import random
from PIL import Image
from deepface import DeepFace
#from huggingface_hub import login

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Argument parser
parser = argparse.ArgumentParser(description="Filter images of pretty women using face embeddings and percentile-based thresholding.")
parser.add_argument("--input", required=True, help="Input folder of images")
parser.add_argument("--output", required=True, help="Folder to save filtered original images")
parser.add_argument("--ref", required=False, help="Folder containing reference model-quality images", default="./reference_models")
parser.add_argument("--pt", required=False, help="Percentile threshold (e.g. 10 for top 10%%)", default="10")
parser.add_argument("--token", required=False, help="Hugging Face token if needed")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to process")
parser.add_argument("--sample", type=int, default=100, help="Number of random samples for baseline scoring")
args = parser.parse_args()

# Optional Hugging Face login
#if args.token:
#    login(token=args.token)

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)

# Get list of already processed filenames
already_processed = set(os.listdir(args.output))

# Load reference embeddings from reference image folder
def load_reference_embeddings(ref_folder):
    embeddings = []
    for fname in os.listdir(ref_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(ref_folder, fname)
            try:
                reps = DeepFace.represent(img_path=fpath, model_name="Facenet", enforce_detection=False)
                if reps:
                    embeddings.append(np.array(reps[0]["embedding"]))
                    if args.debug:
                        print(f"Loaded reference: {fname}")
            except Exception as e:
                print(f"⚠️ Failed to load reference {fname}: {e}")
    return embeddings

reference_embeddings = load_reference_embeddings(args.ref)
if not reference_embeddings:
    print(f"❌ No valid reference embeddings found in '{args.ref}'")
    exit(1)

# Cosine similarity between two vectors
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# Average similarity across multiple reference embeddings
def average_similarity(face_embedding, references):
    scores = [cosine_similarity(face_embedding, ref) for ref in references]
    return sum(scores) / len(scores)

# Check for single face and get embedding
def get_face_embedding(image_path):
    try:
        reps = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        if len(reps) == 1:
            return np.array(reps[0]["embedding"])
    except Exception as e:
        print(f"Face embedding error in {image_path}: {e}")
    return None

# Calculate similarity scores for a random sample to estimate baseline
image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input)
               if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in already_processed]
random.shuffle(image_paths)
sample_paths = image_paths[:min(args.sample, len(image_paths))]
sample_scores = []

for spath in sample_paths:
    emb = get_face_embedding(spath)
    if emb is not None:
        sim = average_similarity(emb, reference_embeddings)
        sample_scores.append(sim)

if not sample_scores:
    print("❌ Could not extract baseline from sample. Exiting.")
    exit(1)

baseline_cutoff = np.percentile(sample_scores, 100 - float(args.pt))
if args.debug:
    print(f"Sample Score Range: Min={min(sample_scores):.4f} Max={max(sample_scores):.4f}")
    print(f"✅ Using percentile threshold {args.pt}% -> Cutoff: {baseline_cutoff:.4f}")

# Main filtering pass
processed = 0
for fpath in image_paths:
    if args.limit and processed >= args.limit:
        break
    fname = os.path.basename(fpath)
    emb = get_face_embedding(fpath)
    if emb is not None:
        sim = average_similarity(emb, reference_embeddings)        
        if sim >= baseline_cutoff:
            print(f"✅ PASS: {fname} | Similarity: {sim:.4f}")
            shutil.copy2(fpath, os.path.join(args.output, fname))
            processed += 1
        else:
            print(f"⚠️ LOW SCORE: {fname} | Similarity: {sim:.4f}")
    else:
        print(f"❌ NO SINGLE FACE: {fname}")
