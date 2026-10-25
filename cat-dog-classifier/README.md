# Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN) project that classifies uploaded images as cats or dogs using a trained deep learning model.

## 🚀 Live Demo

**Try the website here:** [Cat vs Dog CNN Classifier](https://aiml-practice-ck23udc7nuv57n2fdmgmec.streamlit.app/)

Upload a cat or dog image and view the model's prediction.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow

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
## Testing and Sample Results

I tested the Cat vs Dog CNN Classifier using three sample images.

| Test image | Model prediction | Model score | Result |
|---|---|---:|---|
| Cat photo (`my_pet.jpeg`) | Cat | 99.2% | Correct |
| Dog photo (`dog1.jpeg`) | Dog | 99.4% | Correct |
| Unrelated personal photo | Dog | 68.4% | Incorrect / unrelated image |

These are results from only three sample tests. They do not represent the model's overall accuracy.

## Robustness Testing

To test the classifier, try the following types of images:

1. A clear photo of a cat.
2. A clear photo of a dog.
3. An image containing neither a cat nor a dog.
4. A blurry or low-quality image.
5. An image containing multiple animals.

Observe whether the predictions are correct and whether the confidence warning appears.

## Limitations

- The model is designed to classify images into only two classes: Cat or Dog.
- It may classify unrelated images as either a cat or a dog.
- A high model score does not guarantee a correct prediction.
- The confidence score is not a calibrated probability.
- The low-confidence warning is a basic threshold-based warning, not a reliable unrelated-image detector.

## Future Improvements

- Add a method to detect images that do not contain cats or dogs.
- Evaluate the model on a larger, separate test dataset.
- Test different lighting conditions, backgrounds, and image quality.
- Improve the model using more training data and suitable model architectures.
## Model Evaluation

The saved CNN model was evaluated on 4,652 images from the validation split of the Cats vs Dogs dataset.

| Metric | Result |
|---|---:|
| Validation accuracy | 83.75% |
| Precision (Dog class) | 83.92% |
| Recall (Dog class) | 84.27% |
| F1-score (Dog class) | 84.10% |
| Images evaluated | 4,652 |

### Confusion Matrix

| Actual class | Predicted Cat | Predicted Dog |
|---|---:|---:|
| Cat | 1,897 | 383 |
| Dog | 373 | 1,999 |

The model correctly classified 3,896 of the 4,652 validation images. These results measure performance on the validation split, not on an independent final test dataset. Performance may vary on images from different sources or conditions.
