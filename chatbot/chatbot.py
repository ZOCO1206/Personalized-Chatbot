import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmetizer = WordNetLemmatizer()
intents = json.loads(open('E:\\Codes\\Python\\chatbot\\intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('E:\\Codes\\chatbot_model.h5')

def clean_up_scentence(scentence):
    scentence_words =nltk.words = nltk.word_tokenize(scentence)
    scentence_words = [lemmetizer.lemmatize(word) for word in scentence_words]
    return scentence_words

def bag_of_words(scentence):
    scentence_words = clean_up_scentence(scentence)
    bag = [0] * len(words)
    for w in scentence_words:
        for i, word in enumerate(words):
            if word ==w:
                bag[i] = 1
    return np.array(bag)

#defining predict function 
def predict_class(scentence):
    bow = bag_of_words(scentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probabliti': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random. choice (i['responses'])
            break
    return result
print("GO! Bot is running!")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)