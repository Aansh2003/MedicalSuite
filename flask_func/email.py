import smtplib, ssl
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import cv2 
import os

def generate_email(name,prob,pred,type):
    html = """\
    <html>
    <body style="background-color:white;">
        <p>Hi %s,<br>
        Your result = %s<br>
        Probability = %s%%<br>
        Thank you for using our %s<br>
        Be sure to provide feedback at our contact page.<br>
        Thanks,<br>
        Our team
        </p>
    </body>
    </html>
    """%(name.capitalize(),pred,prob,type)
    return html

def send_email(receiver,html, img):
    creds = open('flask_func/login_credentials.txt','r')
    credentials = creds.read().split()

    port = 465
    smtp_server = "smtp.gmail.com"
    sender_email = credentials[0]
    receiver_email = receiver
    password = credentials[1]

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Medical Analysis"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    part = MIMEText(html, 'html')
    msg.attach(part)

    temp_path = "temp.jpg"

    cv2.imwrite(temp_path, img) 
    attachment = open(temp_path, "rb")
    os.remove(temp_path)
    filepath = 'upload' + ".jpg"
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filepath)
    msg.attach(p)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email,msg.as_string())