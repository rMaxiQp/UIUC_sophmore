from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <html>
        <h1>Hello, Cloud!</h1>
        <p>This is a simple web server built with Flask for CS398 MP7</p>
    <html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)