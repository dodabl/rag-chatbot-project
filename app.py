from flask import Flask, render_template
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html", person_name=None)
@app.route("/<person_name>")
def get_profile(person_name: str):
    return render_template("index.html", person_name=person_name)