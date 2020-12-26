from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

flask_app = Flask(__name__)


def xgb_eval_dev_gamma(yhat, dtrain):
    y = dtrain.get_label()
    return 'dev_gamma', 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)


model_avgclaim = xgb.XGBRegressor()
model_avgclaim.load_model('avg_claims_model.pkl')
model_claimcounts = pickle.load(open('claim_counts_model.pkl', 'rb'))


@flask_app.route('/')
def index():
    return 'Hello World!'


@flask_app.route('/predict', methods=['POST'])
def predict():
    request_json = request.json
    id_ = request_json['id']
    df_for_response = pd.DataFrame(data=[[request_json['Exposure'],
                                          request_json['Gender'],
                                          request_json['MariStat'],
                                          request_json['BonusMalus'],
                                          request_json['DrivAgeSq'],
                                          request_json['LicAgeYr']]],
                                   columns=['Exposure', 'Gender', 'MariStat', 'BonusMalus', 'DrivAgeSq', 'LicAgeYr'])

    prediction_avgclaim = model_avgclaim.predict(df_for_response)
    prediction_claimcounts = model_claimcounts.predict(df_for_response)
    claimamount = prediction_avgclaim * prediction_claimcounts

    result = {'id': id_,
              'value_avgclaim': str(prediction_avgclaim[0].round(2)),
              'value_claimcounts': str(int(prediction_claimcounts[0])),
              'value_claimamount': str(claimamount[0].round(2))}

    return jsonify(result)


if __name__ == '__main__':
    flask_app.run(debug=True)
