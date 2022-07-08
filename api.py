from copyreg import pickle
from flask import Flask, request
import joblib
import numpy as np 
import sys 
app = Flask(__name__)
app.config["DEBUG"] = True

# file path to the saved model
model_filepath = 'model.sav'
try:
    global model
    model = joblib.load(open(model_filepath,  'rb'))
    
except: 
    sys.exit('Unable to load to the model')

@app.route('/predict', methods=['GET'])
def predict():    
    try:
        # Get parameters for depression
        q_1 = int(request.args.get('could_not_experience_the_positive_feeling'))
        q_2 = int(request.args.get('could_not_work_up_the_initiative_to_do_things'))
        q_3 = int(request.args.get('had_nothing_to_look_forward_to'))
        q_4 = int(request.args.get('felt_down_hearted_and_blue'))
        q_5 = int(request.args.get('were_unable_to_become_enthusiastic'))
        q_6 = int(request.args.get('felt_you_werent_worth_much_as_a_person'))
        q_7 = int(request.args.get('felt_life_was_meaningless'))
        
        # Predict depression level
        # Same order as the x_train dataframe
        features = [np.array([q_1, q_2, q_3, q_4, q_5, q_6, q_7])]
        
        prediction = model.predict(features)
        output = round(prediction[0][0], 2)
        return {'score': output}
    
    except Exception as e:
        print(e)
        return 'Calculation Error', 500
    

app.run()
