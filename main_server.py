from flask import Flask,render_template,request,render_template_string,redirect,flash, session
from flask_session import Session
import flask_func.retrieve as retrieve
from flask_func.email import *
import flask_func.model_inference as model
import pyrebase
from firebase_func.cloud import *

app = Flask(__name__)
app.secret_key = "super secret key"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

with open('firebase_func/credentials.txt', 'r') as f:
    creds = [line.strip() for line in f]
config = {
    "apiKey": creds[0],
    "authDomain": creds[1],
    "databaseURL": creds[2],
    "storageBucket": creds[3]
}
firebase = pyrebase.initialize_app(config)

@app.route("/", methods=["POST", "GET"])
def index():
    if session.get("id"):
        return redirect("/home")
    if request.method == 'POST':
        if 'email_l' in request.form:
            username = request.form['email_l']
            password = request.form['password_l']
            var = check_pass(username,password,firebase)
            if type(var) == type(''):
                return render_template('login.html',message='Incorrect password or email')
            else:
                session["id"] = var[1]
                session['token'] = var[0]
                return redirect('/home')
        else:
            if request.form['password_s'] != request.form['conf_password_s']:
                return render_template('login.html', message='Passwords don\'t match')
            if len(request.form['password_s']) < 6:
                return render_template('login.html', message='Use a password with more than 6 characters')
            username = request.form['username_s']
            password = request.form['password_s']
            phone = request.form['phone_s']
            gender = request.form['gender_s']
            age = request.form['age_s']
            email = request.form['email_s']
            var = signup(username,email,phone,gender,age,password,firebase)
            if type(var) == type(''):
                return render_template('login.html',message=var)
            else:
                return render_template('login.html',message1='Signup successful')
    return render_template('login.html')
 

@app.route("/profile")
def profile():
    if not session.get("id"):
        return redirect("/")
    uid = session['id']
    data = get_user_data(uid,firebase)
    print(data)
    return render_template("profile.html",age=data['age'],email=data['email'],sex=data['gender'],username=data['name'],phone=data['phone'])
#OrderedDict([('age', '19'), ('email', 'aansh.basu@gmail.com'), ('gender', 'M'), ('name', 'aanshbasu'), ('phone', '8886783089')])

@app.route("/faq")
def faq():
    if not session.get("id"):
        return redirect("/")
    return render_template("faq.html")


@app.route("/logout")
def logout():
    session["id"] = None
    return redirect("/")

@app.route('/home')
def render_home_page():
    if not session.get("id"):
        return redirect("/")
    return render_template('index.html')

@app.route('/tumor_detect',methods=['POST','GET'])
def render_tumor_detect():
    if not session.get("id"):
        return redirect("/")
    if request.method == 'POST':
        name,email,image = retrieve.validate(request)
        prob,pred,segment = model.brainTumor(image)
        pred = model.brain_tumor_parse(pred)
        html = generate_email(name,prob,pred,'Brain Tumor Detect')
        send_email(email,html,segment)
        send_result_data(session['id'], image, segment,pred, firebase)
        flash("True")
        return render_template('BrainTumor.html',email=email)
    return render_template('BrainTumor.html')

@app.route('/retinal_detect',methods=['POST','GET'])
def render_retinal_detect():
    if not session.get("id"):
        return redirect("/")
    if request.method == 'POST':
        name,email,image = retrieve.validate(request)
        prob,pred,segment = model.retinal_scan(image)
        pred = model.retinal_parse(pred)
        html = generate_email(name,prob,pred,'Retinal Disease Detect')
        send_email(email,html,segment)
        send_result_data(session['id'],image,segment,pred,firebase)
        flash("True")
        return render_template('Retinal.html',email=email)
    return render_template('Retinal.html')

@app.route('/info')
def render_info():
    if not session.get("id"):
        return redirect("/")
    return render_template('information.html')

@app.route('/feedback',methods=['POST','GET'])
def render_feedback():
    if not session.get("id"):
        return redirect("/")
    if request.method == 'POST':
        feedback(session['id'],request.form['name'],request.form['email'],request.form['message1'],request.form['message2'],request.form['message3'],firebase)
    return render_template('contact.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)