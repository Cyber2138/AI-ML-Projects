from flask import Flask, render_template, request, redirect
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

model_path = 'C:\\Users\\admin dell\\Desktop\\Completed projects\\MIlitary Defense\\Dynamic\\model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form_page():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    success_rate = request.form["Success Rate"]
    Resilience_Index = request.form["Resilience Index"]
    Communication_Quality = request.form["Communication Quality"]
    Morale_Score = request.form["Morale Score"]
    Casualty_Count = request.form["Casualty Count"]
    Intel_Availability = request.form["Intel Availability"]
    
    if (Intel_Availability == 'CMB'):
        Intel_Availability=0
    elif( Intel_Availability=='TRN'):
        Intel_Availability=1
    elif( Intel_Availability=='REC'):
        Intel_Availability=2

    Training_Level = request.form["Training Level"]
    if (Training_Level == 'ARM'):
        Training_Level=0
    elif( Training_Level=='AVI'):
        Training_Level=1
    elif( Training_Level=='UKN'):
        Training_Level=2
   
    Supply_Level = request.form["Supply Level"]
    if (Supply_Level == 'URB'):
        Supply_Level=0
    elif( Supply_Level=='RUR'):
        Supply_Level=1
    elif( Supply_Level=='COS'):
        Supply_Level=2

    Engagement_Frequency = request.form['Engagement Frequency']
    if (Engagement_Frequency=='RAR'):
        Engagement_Frequency=0
    elif(Engagement_Frequency=='FRQ'):
        Engagement_Frequency=1
    elif(Engagement_Frequency=='OCS'):
        Engagement_Frequency=2

    t = [[float(success_rate), float(Resilience_Index), float(Communication_Quality),float(Morale_Score), float(Casualty_Count), float(Engagement_Frequency),float(Intel_Availability), float(Supply_Level), float(Training_Level)]]

    output = model.predict(t)

    if output[0]=='Successful':
        return render_template('submit.html', prediction_value = 'The Mission is Successful')

    if output[0]=='Ongoing':
        return render_template('submit.html', prediction_value = 'The Mission is Ongoing')

    if output[0]=='Failed':
        return render_template('submit.html', prediction_value = 'The Mission is Failed')
  


if __name__ == '__main__':
    app.run(debug=True)
