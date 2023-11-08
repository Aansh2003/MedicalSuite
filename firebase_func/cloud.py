import pyrebase 
import cv2
import base64
import json
import pickle
from PIL import Image
import numpy

def im2json(im):
    imdata = pickle.dumps(im)
    jstr = json.dumps({"image": base64.b64encode(imdata).decode('ascii')})
    return jstr

def json2im(jstr):
    load = json.loads(jstr)
    imdata = base64.b64decode(load['image'])
    im = pickle.loads(imdata)
    return im

def check_pass(email,password,firebase):
    auth = firebase.auth()
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        uid = auth.get_account_info(user['idToken'])['users'][0]['localId']
        return user['idToken'], uid
    except Exception as e:
        return json.loads(e.args[1])["error"]["message"]

def signup(name, email, phone, gender, age, password, firebase):
    auth = firebase.auth()
    try:
        auth.create_user_with_email_and_password(email, password)
        user = auth.sign_in_with_email_and_password(email, password)
        id = auth.get_account_info(user['idToken'])['users'][0]['localId']
        data = {
        "name": name,
        "email": email,
        "phone": phone,
        "gender": gender,
        "age": age
        }
        db = firebase.database()
        #db.child("users")
        db.child("users").child(id).set(data)
        return True
    except Exception as e:
        
        return json.loads(e.args[1])["error"]["message"]

def get_user_data(id, firebase):
    db = firebase.database()
    try:
        data = db.child("users").child(id).get()
        return data.val()
    except Exception as e:
        return json.loads(e.args[1])["error"]["message"]

def feedback(id, name, email, improvement, experience, etc, firebase):
    db = firebase.database()
    try:
        data = {
        "name": name,
        "email": email,
        "improvement": improvement,
        "experience": experience,
        "etc": etc
        }
        db.child("feedback").child(id).push(data)
        return True
    except Exception as e:
        return json.loads(e.args[1])["error"]["message"]

def send_result_data(id, origin, result, pred,firebase):
    db = firebase.database()
    origin = im2json(origin)
    result = im2json(result)
    try:
        data = {
        "origin": origin,
        "result": result,
        "prediction": pred
        }
        db.child("results").child(id).push(data)
        return True
    except Exception as e:
        return json.loads(e.args[1])["error"]["message"]

def get_result_data(id, firebase):
    db = firebase.database()
    try:
        results = db.child("results").child(id).get()
        origin_result = []
        inference_result = []
        for result in results.each():
            origin_result.append(json2im(result.val()["origin"]))
            inference_result.append(json2im(result.val()["result"]))
        return origin_result, inference_result        
    except Exception as e:
        return json.loads(e.args[1])["error"]["message"]
    







