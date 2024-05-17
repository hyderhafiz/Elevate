from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, validators

class LoginForm(FlaskForm):
    email = StringField('Email', [validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', [validators.DataRequired()])
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    full_name = StringField('Full Name', [validators.DataRequired()])
    username = StringField('Username', [validators.DataRequired()])
    email = StringField('Email', [validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', [validators.DataRequired(), validators.EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Repeat Password')
    submit = SubmitField('Sign Up')
