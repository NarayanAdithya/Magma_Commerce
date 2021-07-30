from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import config




app=Flask(__name__)
socketio=SocketIO(app)
app.config.from_object(config)
db=SQLAlchemy(app)
migrate=Migrate(app,db)
login=LoginManager(app)

login.login_view='login'
login.message='Please Login First'
login.login_message_category='info'

from app import routes,models
