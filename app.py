from flask import Flask, render_template, send_file
import os

app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/frame")
def frame():
    # Send the latest detection image
    return send_file(
        "static/LanternFly_Snapshot_Box.png",
        mimetype="image/png"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)