# Emotion Detection of text
Through this project we will try to predict the emotion of the given statment. In this project I have trained the data using Logistic Regression and Naive Bayes as we have to classify the text if it revels joy, guilt, anger, saddness, shame, disgust or fear.

Within st_model.py, I have used streamlit to expalin the model accuracy in the web page and saved the model using joblib for further use.

Within server.py, I have loaded the saved models and have used FastAPI to predict and pass the result while called.

Within test_emotion.py, We can choose the model(Logistic or Naive) and pass the statment whose emotion is to be detected to server.py through predict button.


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

```
3. Activate the environment
```Linux
  source env/bin/activate (for linux)
```Window
  .\env\Scripts\activate (for window)
```

4. Install the required packages

```python3
  pip3 install -r requirements.txt

``` 

5. Run the UI to know about Model and its accuracy
```python3
  streamlit run model_st.py
```

6. Run the server
```python3
  python3 main.py

```

7. Run the UI to predict the sentiment of text
```python3
  streamlit run predict.py
```