from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)

@app.route('/', methods=['post', 'get'])
def main():
    if request.method == 'POST':
        if request.form['button'] == 'analys':
            ol_code = 
            cl = 
            n = 
            v = 

