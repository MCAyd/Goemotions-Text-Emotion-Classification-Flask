from flask import Flask, request, render_template, redirect, url_for, session
import secrets
from BERTClass import BERTClass
import transformers
import torch
import datetime
import torchtext
import forms
import numpy as np

torch.backends.cudnn.deterministic = True
print("PyTorch Version: ", torch.__version__)
print("torchtext Version: ", torchtext.__version__)

model = torch.load('saved_model', map_location=torch.device('cpu'))

def requestResults(content):
	results, thr_results = model.predict(content)
	return results.tolist(), thr_results.tolist()

app = Flask(__name__)
SECRET_KEY = secrets.token_urlsafe(16)
app.config['SECRET_KEY'] = SECRET_KEY

@app.route('/', methods=['POST', 'GET'])
def post_input():
	session['class_names'] = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'confusion', 
	'curiosity','disapproval', 'gratitude', 'joy', 'love', 'optimism', 'sadness', 'neutral', 'other_emotions']
	session['results'] = np.zeros(14).tolist()
	session['thr_results'] = ''

	form = forms.InputForm(request.form)
	if request.method == "POST":
		content = request.form.get('content') ##if request from terminal or LOCUST
		if form.validate_on_submit():
			content = form.content.data
			session['content'] = content
		# print(content)
		results, thr_results = requestResults(content)
		session['results'] = [round(float(prob), 4) for prob in results[0]]
		session['thr_results'] = thr_results[0]

		result_dict = {}
		for class_name, result in zip(session['class_names'], session['results']):
			result_dict[class_name] = result

		sorted_result_dict = dict(sorted(result_dict.items(), key=lambda x: x[1], reverse=True))
		session['class_names'] = list(sorted_result_dict.keys())
		session['results'] = list(sorted_result_dict.values())
		# print(sorted_result_dict)

		if form.validate_on_submit(): ##if user gives input on UI
			log_text = f"User input: {content}\n"
			log_text += f"Sorted results: {sorted_result_dict}\n"
			log_text += f"Timestamp: {datetime.datetime.now()}\n\n"
			with open("log.txt", "a") as log_file:
				log_file.write(log_text)
			return render_template('home.html', form=form)

		return sorted_result_dict ##if user gives input on terminal

	return render_template('home.html', form=form)

@app.route('/result', methods=['POST', 'GET'])
def get_result():

	class_probabilities = session['results']

	return render_template('result.html', class_probabilities=class_probabilities)

if __name__ == '__main__' :
    app.run(debug=True)