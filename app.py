from flask import Flask, render_template
from flask import request
import main as m
app=Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def result():
    output=request.form.to_dict()
    x=m.recommend(output)
    name=x['name']
    
    return render_template('index.html', name= name)
if __name__ == '__main__':
    app.run(debug=True)
