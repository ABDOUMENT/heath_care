import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import wolframalpha
import torch
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()

# Label encoding
y_encoded = preprocessing.LabelEncoder().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
testx = testing[cols]
testy = preprocessing.LabelEncoder().fit(y).transform(testing['prognosis'])

# Train models
clf = DecisionTreeClassifier().fit(x_train, y_train)
print("Decision Tree Accuracy:", cross_val_score(clf, x_test, y_test, cv=3).mean())

model = SVC().fit(x_train, y_train)
print("SVM Accuracy:", model.score(x_test, y_test))

# WolframAlpha Client
wolfram_client = wolframalpha.Client("K4Q6JX-TKXX2PKR8R")  # Replace with your API Key

# Global dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

# Load sentence transformer model and symptom embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
symptom_list = list(symptoms_dict.keys())
symptom_embeddings = embed_model.encode(symptom_list, convert_to_tensor=True)

# Text to speech function
def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

# Load dictionaries
def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue
            try:
                severityDictionary[row[0].strip()] = int(row[1])
            except ValueError:
                continue

def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:
                continue
            description_list[row[0]] = row[1]

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 5:
                continue
            precautionDictionary[row[0]] = row[1:5]

# Semantic symptom matcher
def check_pattern(dis_list, inp):
    inp_embedding = embed_model.encode(inp, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(inp_embedding, symptom_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=3)
    threshold = 0.4
    pred_list = []
    for score, idx in zip(top_results.values, top_results.indices):
        if score > threshold:
            pred_list.append(symptom_list[idx])
    return (1, pred_list) if pred_list else (0, [])

# Secondary prediction
def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    rf_clf = DecisionTreeClassifier().fit(X, y)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])

# Print disease
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = preprocessing.LabelEncoder().fit(y).inverse_transform(val[0])
    return list(map(str.strip, list(disease)))

# Severity calculator
def calc_condition(exp, days):
    score = sum(severityDictionary.get(item, 0) for item in exp)
    if (score * days) / (len(exp) + 1) > 13:
        print("\nYou should consult a doctor.")
    else:
        print("\nIt might not be serious, but precautions are recommended.")

# Start interaction
def main_loop():
    print("-----------------------------------HealthCare ChatBot with WolframAlpha-----------------------------------")
    print("Hi! I'm Mana, your healthcare assistant.")
    print("Ask your question or describe your symptoms:")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Stay healthy.")
            break

        # Try WolframAlpha first
        try:
            res = wolfram_client.query(user_input)
            answer = next(res.results).text
            print(answer)
            continue
        except Exception:
            pass

        # Healthcare assistant
        try:
            symptom_inputs = [user_input.lower()]
            chk_dis = ",".join(cols).split(",")
            confirmed_symptoms = []

            for symptom in symptom_inputs:
                conf, cnf_dis = check_pattern(chk_dis, symptom)
                if conf:
                    print(f"Related symptoms for '{symptom}':")
                    for num, it in enumerate(cnf_dis):
                        print(num, ")", it)
                    selected = 0 if len(cnf_dis) == 1 else int(input(f"Select the one you meant (0 - {len(cnf_dis)-1}): "))
                    confirmed_symptoms.append(cnf_dis[selected])
                else:
                    print(f"im a healthcare chat bot , you can give me symptoms and i can help")

            if not confirmed_symptoms:
                continue

            num_days = int(input("For how many days have you experienced these symptoms? "))

            def recurse(node, depth):
                tree_ = clf.tree_
                feature_name = [cols[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
                symptoms_present = []

                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    val = 1 if name in confirmed_symptoms else 0
                    if val <= threshold:
                        recurse(tree_.children_left[node], depth + 1)
                    else:
                        symptoms_present.append(name)
                        recurse(tree_.children_right[node], depth + 1)
                else:
                    present_disease = print_disease(tree_.value[node])
                    symptoms_given = reduced_data.columns[reduced_data.loc[present_disease].values[0].nonzero()]
                    symptoms_exp = []
                    for syms in symptoms_given:
                        resp = input(f"Are you experiencing {syms.replace('_', ' ')}? (yes/no): ").strip().lower()
                        if resp == "yes":
                            symptoms_exp.append(syms)

                    second_prediction = sec_predict(symptoms_exp)
                    calc_condition(symptoms_exp, num_days)

                    print("\nPossible diagnosis:")
                    if present_disease[0] == second_prediction[0]:
                        print("You may have:", present_disease[0])
                        print(description_list.get(present_disease[0], "No description available."))
                    else:
                        print("You may have:", present_disease[0], "or", second_prediction[0])
                        print(description_list.get(present_disease[0], "No description available."))
                        print(description_list.get(second_prediction[0], "No description available."))

                    precautions = precautionDictionary.get(present_disease[0], [])
                    print("\nTake the following precautions:")
                    for i, p in enumerate(precautions):
                        print(i + 1, ")", p)

            recurse(0, 1)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try again.\n")

# Initialize dictionaries and start chatbot
getSeverityDict()
getDescription()
getprecautionDict()
main_loop()
