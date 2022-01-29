from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pickle as p
import json
import os
import sys

from LogRegWrapper import LogRegWrapper
from DecisionTreeWrapper import DecisionTreeWrapper


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin()
def makecalc():
    print("hmm")
    data = request.get_json()
    logPred = LogRegWrapper().predict(data)
    decPred = DecisionTreeWrapper().predict(data)
    response = jsonify(
        logPred = logPred,
        DecPred = decPred
    )
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    # os.system('python LogRegWrapper.py')
    # os.system('python DecisionTreeWrapper.py')
    app.run(threaded=True, port=5000)