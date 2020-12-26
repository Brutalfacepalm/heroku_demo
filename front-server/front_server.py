import json

from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired, Length, NumberRange
import requests


class ClientDataForm(FlaskForm):
    id = IntegerField('ID', validators=[DataRequired()])
    exposure = IntegerField('Срок действия полиса(мес.)', validators=[DataRequired()])
    age = IntegerField('Возраст водителя', validators=[DataRequired(),
                                                       NumberRange(min=18, max=110, message="Возраст от 18 до 110")])
    gender = SelectField('Пол', choices=[(1, 'М'), (0, 'Ж')])
    mari_stat = SelectField('Семейное положение', choices=[(0, 'Alone'), (1, 'Other')])
    lic_age = IntegerField('Водительский стаж',
                           validators=[DataRequired(),
                                       NumberRange(min=18, max=110, message="Возраст от 18 до 110")])
    bonus_malus = SelectField('Класс КБМ', choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5'),
                                                    (6, '6'), (7, '7'), (8, '8'), (9, '9'), (10, '10'), (11, '11'),
                                                    (12, '12'), (13, '13')])


def send_json(data):
    url = 'https://insurance-made.herokuapp.com/predict'
    headers = {'content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm(request.form)
    print(form.validate())

    data = {}
    if request.method == 'POST':
        data['id'] = request.form.get('id')
        data['DrivAgeSq'] = float(request.form.get('age'))**2
        data['LicAgeYr'] = float(request.form.get('lic_age'))
        data['Gender'] = float(request.form.get('gender'))
        data['MariStat'] = float(request.form.get('mari_stat'))
        data['Exposure'] = float(request.form.get('exposure'))/12
        data['BonusMalus'] = float(request.form.get('bonus_malus'))
        try:
            response = send_json(data)
            response = response.text
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='127.0.0.2', port=5000)
