import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import numpy as np
import pickle
filename = 'PREDICTOR.sav'
PREDICTOR = pickle.load(open(filename, 'rb'))

#-------- ROUTES GO HERE -----------#
@app.route('/page')
def page():
   with open("titanic_entry.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       pclass = inputs['pclass'][0]
       sex = inputs['sex'][0]
       age = inputs['age'][0]
       fare = inputs['fare'][0]
       sibsp = inputs['sibsp'][0]

       item = np.array([pclass, sex, age, fare, sibsp]).reshape(1,-1)
       score = PREDICTOR.predict_proba(item)
       results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       return flask.jsonify(results)
	   
if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4000

    app.run(HOST, PORT)