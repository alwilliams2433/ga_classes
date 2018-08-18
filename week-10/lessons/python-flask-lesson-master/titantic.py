import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import numpy as np
import pickle
filename = 'PREDICTOR.sav'
PREDICTOR = pickle.load(open(filename, 'rb'))

#-------- ROUTES GO HERE -----------#
@app.route('/predict', methods=["GET"])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']

    item = np.array([pclass, sex, age, fare, sibsp]).reshape(1,-1)
    score = PREDICTOR.predict_proba(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)
	
	
if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4000

    app.run(HOST, PORT)