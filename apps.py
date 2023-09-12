from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import requests

model = pickle.load(open('./model/AirlinePassenger.pkl', 'rb'))
app = Flask (__name__, template_folder='templates/')
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/pred', methods=['GET', 'POST']) 
def predict():
    if request.method == "POST":
        age = request.form.get("age")
        dist = request.form.get("dist")
        wifi = request.form.get("wifi")
        arrival = request.form.get("arrival")
        onlinebooking = request.form.get("onlinebooking")
        location = request.form.get("location")
        food = request.form.get("food")
        onlineboarding = request.form.get("onlineboarding")
        seat = request.form.get("seat")
        inflightenv = request.form.get("inflightenv")
        onboardservice = request.form.get("onboardservice")
        roomservice = request.form.get("roomservice")
        handling = request.form.get("handling")
        checkin = request.form.get("checkin")
        inflight = request.form.get("inflight")
        cleanliness = request.form.get("cleanliness")
        delay = request.form.get("delay")
        arivaldelay = request.form.get("arivaldelay")
        
        
        total = [[age, dist, wifi, arrival, onlinebooking, location, food, onlineboarding, seat, inflightenv, onboardservice, roomservice, handling, checkin, inflight, cleanliness, delay, arivaldelay]]
        
        model = joblib.load("model/AirlinePassenger.pkl")
        
                
        pred = model.predict(total)
       
        if int(pred) == 0:
            pre = "Passengers have satisfies the Airline Service"
        else:
            pre = "Passengers have neutral or dissatisfied the Airline Service"

        return render_template('predict.html', prediction_text=pre)
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)

