from werkzeug.utils import secure_filename
import numpy as np
import cv2

def retrieve_values(server_dictionary):
    name = server_dictionary['name']
    email = server_dictionary['email']
    return name,email

def file_check(file):
    return '.' in file and \
        file.rsplit('.', 1)[1].lower() in {'png','jpg','jpeg'}

def validate(request):
    name,email = retrieve_values(request.form)
    file = request.files['file']
    if not file_check(file.filename):
        raise Exception('Incorrect file format')
    content = file.read()
    nparr = np.fromstring(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return name,email,image