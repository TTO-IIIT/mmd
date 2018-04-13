# making api of the mmd
import test_mmd
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():

   return ("hello hello hello")

if __name__ == "__main__":
        app.run()
