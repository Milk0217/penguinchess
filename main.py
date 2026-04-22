from flask import Flask, send_from_directory

app = Flask(__name__, static_url_path='', static_folder='.')

@app.route('/')
def home():
    return send_from_directory('./templates', 'index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
