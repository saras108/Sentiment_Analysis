# Emotion Detection of text
Through this project we will try to predict the emotion of the given statment. In this project I have trained the data using Logistic Regression and Naive Bayes as we have to classify the text if it revels: **joy** :smiley:, **guilt** :mask:, **anger** :pout: , **saddness** :frowning_face:	, **shame** :pleading_face:, **disgust** :roll_eyes: or **fear** :weary: .

## Installation

To run this code you would need:

1. Download/ Clone the project

```git
  git clone https://github.com/saras108/Sentiment_Analysis
```

2. Create a virtual environment

```python3
  python3 -m venv env
```

3. Activate the environment
```
  source env/bin/activate (for linux)
  .\env\Scripts\activate (for window)
```

4. Install the required packages

```python3
  pip3 install -r requirements.txt
``` 

5. Run the UI to know about Model and its accuracy
```python3
  streamlit run st_model.py
```

6. Run the server
```python3
  python3 server.py
```

7. Run the UI to predict the sentiment of text
```python3
  streamlit run test_emotion.py
```

Within st_model.py, I have used streamlit to expalin the model accuracy in the web page and saved the model using joblib for further use.

Within server.py, I have loaded the saved models and have used FastAPI to predict and pass the result while called.

Within test_emotion.py, We can choose the model(Logistic or Naive) and pass the statment whose emotion is to be detected to server.py through predict button in the UI.

For more pre processing steps and other details like confusion matrix, nlp.ipynb  is available within this repo.

### HAVE A HAPPY LEARNING :hugs: :hugs: