from flask import Flask, request, jsonify, render_template
from preprocessor import Preprocessor
from namedEntityRecognition import LocationPredictor
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            tweets = data["tweets"]
            classifier = joblib.load('static/models/multinomialNB.pkl')
            classification = classifier.predict(tweets).tolist()
            if classification[0] == 1:
                nerecognizer = joblib.load('static/models/namedEntityRecognition.pkl')
                location = LocationPredictor.predictLocation(nerecognizer, tweets)
                result = [classification, location]
            else:
                result = [classification]
        except ValueError:
            return jsonify("Error.")

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)