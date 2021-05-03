from flask import Flask, request, render_template
# importing Useful DataStructures
import pandas as pd
import numpy as np

# importing Misc Libraries
import pickle
import warnings
import sqlite3 as sql
warnings.filterwarnings('ignore')
import time


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


def final_function_2(main_preprocessor_data):
    '''
    Function 2 for prediction. This function takes both the Test Point and Target value of that point. It returns
    the prediction along with the metric for the predicted points.

    '''

    # load best model
    # Saving the final model LightGBM as pickle file for the future use in productionizing the model
    with open('Best_lgbm.pkl', 'rb') as fp:
        best_model = pickle.load(fp)

    y_pred_prob = best_model.predict(main_preprocessor_data)

    y_pred = np.ones((len(main_preprocessor_data),), dtype=int)
    for i in range(len(y_pred_prob)):
        if y_pred_prob[i] <= 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    predicted_classes = np.where(y_pred_prob > 0.5, 1, 0)


    return predicted_classes, y_pred_prob



@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [int(x) for x in request.form.values()]
    #df_main = pd.read_csv("All_preprocessed_HomeCredit.csv")
    conn = sql.connect('Home_credit.sqlite')
    df_main= pd.read_sql('SELECT * FROM preprocessed where SK_ID_CURR==' +str(int_features[0]), conn)
    #test_datapoint_func_2 = df_main[df_main['SK_ID_CURR'] == int_features[0]]
    targets_func_2 = df_main.drop('SK_ID_CURR',axis=1)
    output, prob_score = final_function_2(main_preprocessor_data = targets_func_2)
    if int(output) == 0:
        return render_template('index.html',
                               prediction_text='Customer having good Credit history, capabale to repay a debt, with probability of being defaulter is '+ str(round(prob_score[0]*100,2))+ ' percentage.')
    else:
        return render_template('index.html', prediction_text='Customer having bad Credit history, risky customer,with probability of being defaulter is  '+ str(round(prob_score[0]*100,2))+ ' percentage.')


if __name__ == "__main__":
    app.run(debug=True,port=5000)
	#app.run(host='0.0.0.0',port=8000)