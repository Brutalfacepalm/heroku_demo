import json
import requests
import pickle

from flask import Flask, render_template, redirect, url_for, request, jsonify, make_response
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired, NumberRange

import pandas as pd
import numpy as np
import xgboost as xgb


def xgb_eval_dev_gamma(yhat, dtrain):
    y = dtrain.get_label()
    return 'dev_gamma', 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

model_avgclaim = xgb.XGBRegressor()
model_avgclaim.load_model('avg_claims_model.pkl')
model_claimcounts = pickle.load(open('claim_counts_model.pkl', 'rb'))


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


def predict(insurance_data):
    id_ = insurance_data['id']
    df_for_response = pd.DataFrame(data=[[insurance_data['Exposure'],
                                          insurance_data['Gender'],
                                          insurance_data['MariStat'],
                                          insurance_data['BonusMalus'],
                                          insurance_data['DrivAgeSq'],
                                          insurance_data['LicAgeYr']]],
                                   columns=['Exposure', 'Gender', 'MariStat', 'BonusMalus', 'DrivAgeSq', 'LicAgeYr'])

    prediction_avgclaim = model_avgclaim.predict(df_for_response)
    prediction_claimcounts = model_claimcounts.predict(df_for_response)
    claimamount = prediction_avgclaim * prediction_claimcounts

    result = {'id': id_,
              'value_avgclaim': str(prediction_avgclaim[0].round(2)),
              'value_claimcounts': str(int(prediction_claimcounts[0])),
              'value_claimamount': str(claimamount[0].round(2))}

    return json.dumps(result)


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
            response = predict(data)
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='127.0.0.3', debug=True)
