from datetime import datetime
from flask import Flask as Fl, redirect, request, url_for, render_template, session, flash
from datetime import timedelta
from Colors import *
app = Fl(__name__)
app.secret_key = "hello"
app.permanent_session_lifetime = timedelta(minutes=60)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["POST", "GET"])
def login(): 
    if request.method =="POST":
        session.permanent == True
        user = request.form["nm"]
        session["user"] = user
        return redirect(url_for("user"))
    else:
        if "user" in session:
            return redirect(url_for("user"))
        return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/user")
def user():    
    if "user" in session:
        user = session["user"]
        user = user.capitalize()
        return render_template("user.html", name = user)
    else:
        return redirect(url_for("login"))

@app.route('/figure', methods=["POST", "GET"])
def figure():
    if request.method == "POST":
        col_num = request.form["num_range"]
        return col_num
    else:
        return render_template("figure.html",pie = Pie_fig())

@app.route('/Pie')
def Pie():
  return Pie_fig()

if __name__ == "__main__":
    app.run(debug =True)
