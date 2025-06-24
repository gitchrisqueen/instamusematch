# InstaMuseMatch

This project filters a folder of images to retain only photos containing exactly one face that closely matches the facial features of reference "model-quality" images. The filtering is based on facial embedding similarity using DeepFace.

---

## Features

* Filters out photos with multiple faces or no detectable face
* Computes similarity to a set of reference model images
* Uses percentile-based thresholding (e.g. top 10%)
* Avoids re-processing previously filtered images

---

## Requirements

### 📦 Dependencies

* Python 3.7+
* Install dependencies:

```bash
pip install -r requirements.txt
```

#### `requirements.txt`

```txt
numpy
opencv-python
Pillow
deepface
```

---

## Setup Instructions

### Step 1: Get Photos to Filter

Use [Hotlistr](https://github.com/gitchrisqueen/hotlistr) to download photos from Instagram followers:

```bash
git clone https://github.com/gitchrisqueen/hotlistr
cd hotlistr
# Follow the instructions to download profile photos using IG Exporter + Downloader Script
```

Place the downloaded images into a folder (e.g. `IG_username_Followers`).

### Step 2: Gather Reference Model Images

Create a `reference_models/` folder with high-quality headshots of beautiful women. Use Google Image search for terms like:

* `Zendaya Headshot`
* `Margot Robbie Vogue`
* `Taylor Hill portrait`
* `Model face symmetry`

Download JPG or PNG images and save them in that folder.

---

## Usage

### Run the Script

```bash
python filter_photos.py \
  --input ./IG_username_Followers \
  --output ./IG_username_Followers_filtered \
  --ref ./reference_models \
  --pt 10 \
  --sample 100 \
  --debug
```

#### Arguments

* `--input`: Folder of images to process
* `--output`: Folder to save passed images
* `--ref`: Folder of reference images (default: `./reference_models`)
* `--pt`: Percentile threshold (e.g. 10 means top 10%)
* `--sample`: Number of random images to baseline scoring from
* `--debug`: Enable detailed logs

---

## Example Output

```
Sample Score Range: Min=0.0312 Max=0.2239
✅ Using percentile threshold 10% -> Cutoff: 0.1802
✅ PASS: jennylove.jpg | Similarity: 0.1856
⚠️ LOW SCORE: randomuser.jpg | Similarity: 0.0741
❌ NO SINGLE FACE: group_shot.png
```

Only the `PASS` images are copied to the `output` folder.

---


## Use Filtered Results with Hotlistr

Once you've filtered images:

1. Copy the filtered folder path to `hotlistr` project:

   ```bash
   cp -r ./IG_username_Followers_filtered ./hotlistr/final_selection
   ```

2. Run the CSV creator to generate profile data:

   ```bash
   cd hotlistr
   python create_hotlist.py --input ./final_selection --output final_profiles.csv
   ```

This CSV will contain info for your curated, attractiveness-filtered profiles.

---

## License

MIT. Created by [Christopher Queen](https://github.com/gitchrisqueen)
