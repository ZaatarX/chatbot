import json
import random
import tensorflow
import tflearn
import pickle
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    patterns = []
    patterns_tags = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            patterns.append(wrds)
            patterns_tags.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, pattern in enumerate(patterns):
        bag = []
        wrds = [stemmer.stem(w) for w in pattern]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(patterns_tags[x])] = 1

        training.append(bag)
        output.append(output_row)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    training = numpy.array(training)
    output = numpy.array(output)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = i

    return numpy.array(bag)


def chat():
    print("Start talking with the bot...")

    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        responses = "Sorry, I didn't quite catch that. I'm one day old, I'm still learning..."

        if results[results_index] > 0.95:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    responses = random.choice(responses)

        print("Pini, the ChatBot: ", responses)


chat()
