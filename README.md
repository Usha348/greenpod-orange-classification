# Orange Image Classification

## Greenpod Labs – Data Science Assignment 2

This project compares two machine learning approaches for classifying orange images into four categories: blackspot, canker, fresh, and grenning.

## Dataset
- Training images: 991
- Testing images: 99
- Total images: 1,090
- Classes: blackspot, canker, fresh, grenning

The orange dataset was used after confirmation that it was acceptable for the assignment.

## Models
1. **Convolutional Neural Network (CNN)** — test accuracy: 94.95%
2. **Random Forest Classifier** — test accuracy: 91.92%

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Streamlit
- Matplotlib

## Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the CNN:

```bash
python main.py
```

Train and evaluate the Random Forest model:

```bash
python random_forest.py
```

Run the web application:

```bash
streamlit run app.py
```

## Limitations
Performance may vary with different lighting conditions, image quality, backgrounds, and images not represented in the training dataset.

## Future Improvements
- Collect more diverse images.
- Apply data augmentation and tune model parameters.
- Test the models on real-world images.
- Explore transfer learning models such as MobileNetV2.
