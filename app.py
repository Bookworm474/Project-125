from flask import Flask, jsonify, request

#import get_prediction function from classifier.py
from classifier import get_prediction

#initialise app
app = Flask(__name__)

#execute on default route
@app.route("/", methods=["POST"])
def predict_data():
    #request provided image file named 'digit'
    image = request.files.get('digit')
    #generate prediction
    prediction = get_prediction(image)

    #return prediction
    return jsonify({
        "prediction": prediction
    })

#run app
if (__name__ == "__main__"):
    app.run(debug = True)