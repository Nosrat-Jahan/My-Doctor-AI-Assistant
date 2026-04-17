"""
-----------------------------------------------------------------------------
  My Doctor AI Assistant | Healthcare Tech Experiment
  Developer: Nosrat Jahan
  Release Version: 9.9.9
  Created: February 2025  <--- [Archive: Original Build Date]
-----------------------------------------------------------------------------
  DEVELOPER'S NOTE:
  I designed and coded this application back in early 2025 to understand 
  how Machine Learning (specifically Naive Bayes) could be used in basic 
  medical diagnostics. 

  The project focuses on natural language symptom processing, where user 
  inputs are mapped against a JSON-structured medical database. Looking 
  back at this 2025 build, it represents my early exploration into Flask-based 
  AI integrations and clinical data automation.
-----------------------------------------------------------------------------
"""

import json
import os
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

def load_medical_data():
    db_file = 'medical_data.json'
    if not os.path.exists(db_file):
        initial_data = {
            "diseases": {
                "Flu": ["fever cough headache body pain chills", "high temperature sore throat"],
                "Common Cold": ["sneezing runny nose mild fever", "cough congestion cold"],
                "Migraine": ["severe headache nausea light sensitivity", "throbbing head pain"],
                "Food Poisoning": ["vomiting diarrhea stomach pain", "nausea abdominal cramps"],
                "Allergy": ["itching skin rash sneezing", "watery eyes swelling"]
            },
            "medicines": {
                "Flu": {
                    "meds": ["Paracetamol (500mg)", "Ibuprofen (200mg)"],
                    "dose": "1 tablet (500mg) every 6-8 hours",
                    "time": "After meals (Morning/Night)"
                },
                "Common Cold": {
                    "meds": ["Cetirizine (10mg)", "Zinc Lozenges"],
                    "dose": "1 tablet (10mg) per day",
                    "time": "Before bed"
                },
                "Migraine": {
                    "meds": ["Naproxen (250mg)", "Sumatriptan (50mg)"],
                    "dose": "1 tablet at onset of pain",
                    "time": "As needed (Maximum 2 in 24h)"
                },
                "Food Poisoning": {
                    "meds": ["ORS (Oral Rehydration)", "Domperidone (10mg)"],
                    "dose": "ORS after every fluid loss; 1 tablet (10mg) for nausea",
                    "time": "30 mins before meals"
                },
                "Allergy": {
                    "meds": ["Loratadine (10mg)", "Fexofenadine (120mg)"],
                    "dose": "1 tablet (10mg/120mg) daily",
                    "time": "Once a day (Morning or Night)"
                }
            }
        }
        with open(db_file, 'w') as f:
            json.dump(initial_data, f, indent=4)
        return initial_data
    with open(db_file, 'r') as f:
        return json.load(f)

data = load_medical_data()
texts, labels = [], []
for disease, symptoms in data['diseases'].items():
    for s in symptoms:
        texts.append(s)
        labels.append(disease)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("symptoms", "").lower()
    if not user_input:
        return jsonify({"error": "No symptoms provided!"}), 400
    vect = vectorizer.transform([user_input])
    prediction = model.predict(vect)[0]
    info = data['medicines'].get(prediction, {})
    return jsonify({
        "condition": prediction,
        "medicines": info.get("meds", []),
        "dose": info.get("dose", "Consult a physician for dosage"),
        "timing": info.get("time", "Follow doctor's instructions"),
        "disclaimer": "⚠️ EDUCATIONAL PURPOSES ONLY. Consult a doctor before any medication."
    })

if __name__ == "__main__":
    app.run(debug=True)
