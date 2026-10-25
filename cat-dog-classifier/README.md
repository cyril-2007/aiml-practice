# Cat vs Dog CNN Classifier

A beginner-friendly web app for the CNN trained in `train_model.py`. It uses the saved `cat_dog_model.keras` model to classify an uploaded photo as a cat or dog.

## 1. Open a terminal in this folder

```powershell
cd C:\Users\Cyril\BCA-AIML-Journey\aiml-practice\cat-dog-classifier
```

## 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the activation command shown by your Python/IDE setup, or run the next commands with `.\.venv\Scripts\python.exe` instead.

## 3. Install the web-app dependency

```powershell
python -m pip install -r requirements.txt
```

## 4. Start the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`. Open it in your browser, upload a JPG or PNG, then select **Classify image**.

## What the app does

1. Converts the uploaded image to RGB.
2. Resizes it to **128 × 128** pixels.
3. Converts pixels from 0–255 to 0–1.
4. Sends the image to `cat_dog_model.keras`.
5. Displays Cat and Dog model scores and the higher-scoring label.

These preprocessing steps match `predict.py` and `train_model.py`. A score is the model's output, not a guarantee that its choice is correct—especially for blurry, unusual, multi-animal, or non-cat/dog photos.

## Git note

The project `.gitignore` excludes `.venv`, model files, test images, and generated upload folders. Keep `app.py`, `requirements.txt`, `README.md`, and your training/prediction scripts in Git; store large trained models separately (for example, in a release or model-storage service) if you later publish the project.
