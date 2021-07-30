from app import app
from app import app,db,socketio
from flask_socketio import emit,leave_room,join_room
from flask import request,redirect,url_for,render_template,flash,get_flashed_messages,flash,jsonify
from flask_login import current_user,login_user,logout_user,login_required
from app.models import User

from werkzeug.urls import url_parse



@app.route('/')
def home():
    return "Hello"


@app.route('/logout')
def logout():
    db.session.commit()
    logout_user()
    return redirect(url_for('index'))

@app.route('/login',methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    # form=LoginForm()
    # if form.validate_on_submit():
    #     user=User.query.filter_by(email=form.email.data).first()
    #     if user is None or not user.check_password(form.password.data):
    #         flash('Invalid Email or Password',category="danger")
    #         return redirect(url_for('login'))
    #     login_user(user,remember=form.remember_me.data)
    #     next_page=request.args.get('next')
    #     if not next_page or url_parse(next_page).netloc!='':
    #         next_page=url_for('index')
    #     return redirect(next_page)
    # return render_template('signinpage.html',title='SignIn',form=form)
    return 1