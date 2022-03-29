import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    model = pickle.load(open('lr.pickle', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    with open("text.txt", "r") as myfile:
        raw_text = myfile.readlines()
        vectorized_text = vectorizer.transform(raw_text)
    prediction = model.predict(vectorized_text)
    if (prediction[0] == 1):
        print('ПОЗИТИВНО')
    else:
        print('НЕГАТИВНО')
    print('* Заявленное качество модели: accuracy = 0.729, F1_score = 0.74 на тестовом датасете')
