# import random
# import json
# import pickle
# import numpy as np
# import tensorflow as tf
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimizers.legacy import SGD
      

# lemmatizer=WordNetLemmatizer()

# intents = json.loads(open('E:\\Codes\\Python\\chatbot\\intents.json').read())
# #json file is used to set the intents or the hardcoded prompts for the chatbot 
# words =[]                           #empty list for words 
# classes = []                        #Empty list for classes
# documents =[]                       #empty list for the document 
# ignore_letters = ['?','!','.',',']  #ignoring the extra letter or cases in the prompt 

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list,intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# #print(documents)
# words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]  #helps us display the words without the charecter sin ingnored list
# words = sorted(set(words)) # sorts the set of words 

# classes  = sorted(set(classes))

# pickle.dump(words,open('words.pkl', 'wb'))      #save the words in a file 
# pickle.dump(classes,open('classes.pkl', 'wb'))    # save the classe in a file 

# #print(words)

# #machine learning using nural network, We have to show the words as numerical value as the nural network doesnt have words already 
# #this is done using the bag of words method the words and classes will be tokenizes and checked wether they are repeted in the prompt as keywords 
# #

# training = []
# output_empty = [0]*len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words: #for empty list word[]
#         bag.append(1) if word in word_patterns else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(document[1])]=1
#     training.append([bag + output_row])
# #shuffling the output by randomizing
# random.shuffle(training)
# training = np.array(training, dtype=object)

# train_x = list(training[:, 0])
# train_y = list(training[:, len(words):])

# #building nural network 
# model = Sequential()
# model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64,activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax')) 
# sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
# model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size =5, verbose=1)
# model.save('chatbot_model.h5')
# print("Done")         
import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('E:\\Codes\\Python\\chatbot\\intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',',"'"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print('Done')