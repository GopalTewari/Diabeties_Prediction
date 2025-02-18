from wsgiref import simple_server
from flask import Flask, request, app, send_file,render_template
from flask import Response
from flask_cors import CORS
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
sns.set()

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predictRoute():
    try:
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        BMI = float(request.form['BMI'])
        Age = int(request.form['Age'])



        with open("Model/sandardScalar.sav", 'rb') as f:
            scalar = pickle.load(f)

        with open("Model/modelForPrediction.sav", 'rb') as f:
            model = pickle.load(f)
        predict = model.predict(scalar.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]))
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        print('result is        ',result)
        return render_template('results.html',prediction=result)
    except Exception as e:
        print('exception is   ',e)
        return Response(e)
#---------------------------------------------------------------------------------------------------------
@app.route('/csv',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def csv():
    if request.method == 'POST':
        try:
            #reading csv file
            uploaded_file = request.files['upload_file']
            filename = uploaded_file.filename

            #procede only if file is available
            if uploaded_file.filename != '':
                uploaded_file.save(filename)
                data = pd.read_csv(filename)


                # procede only if file is in correct format
                if len(data.columns) == 8:

                    #filling NaN values if present in dataset
                    data['Pregnancies'].fillna(value=round(data['Pregnancies'].mean()), inplace=True)
                    data['Glucose'].fillna(value=round(data['Glucose'].mean()), inplace=True)
                    data['BloodPressure'].fillna(value=round(data['BloodPressure'].mean()), inplace=True)
                    data['SkinThickness'].fillna(value=round(data['SkinThickness'].mean()), inplace=True)
                    data['Insulin'].fillna(value=round(data['Insulin'].mean()), inplace=True)
                    data['BMI'].fillna(value=data['BMI'].mean(), inplace=True)
                    data['DiabetesPedigreeFunction'].fillna(value=data['DiabetesPedigreeFunction'].mean(), inplace=True)
                    data['Age'].fillna(value=round(data['Age'].mean()), inplace=True)

                    # loading the model file from the storage
                    with open("Model/sandardScalar.sav", 'rb') as f:
                        scaler = pickle.load(f)

                    with open("Model/modelForPrediction.sav", 'rb') as f:
                        loaded_model = pickle.load(f)
                    #deleting previous files present in csv_file folder
                    csv_files = './csv_file'
                    list_of_files = os.listdir(csv_files)
                    for csfile in list_of_files:
                        try:
                            os.remove("./csv_file/" + csfile)
                        except Exception as e:
                            print('error in deleting:  ', e)

                    # making prediction
                    prediction = loaded_model.predict(scaler.transform(data))
                    data['Predictions'] = prediction

                    #saving pandas dataframe as a csv file in csv_file folder
                    result_file = './csv_file/result_output_data.csv'
                    data.to_csv(result_file)

                    #plot for prediction analysis
                    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
                    total_pridiction = sns.catplot(x='Predictions', kind='count', data=data)
                    age_relation=sns.catplot(x='Predictions', y='Age', data=data)

                    # deleting previous graph images present in statistics folder
                    image_files = './static/statistics'
                    list_of_files = os.listdir(image_files)
                    for imgfile in list_of_files:
                        try:
                            os.remove("./static/statistics/" + imgfile)
                        except Exception as e:
                            print('error in deleting:  ', e)

                    #save graph in statictics folder inside static
                    output_path_total = './static/statistics/output_prediction.png'
                    output_path_age = './static/statistics/relationship_age.png'
                    total_pridiction.savefig(output_path_total)
                    age_relation.savefig(output_path_age)

                    return render_template('csv.html')

                else:
                    return 'Error: Please Make Sure that csv file is in standard acceptable format,Please go through given Sample csv file format'


            else:
                return 'File Not Found'


        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')





@app.route('/uploadfile',methods=['POST','GET'])  #
@cross_origin()
def uploadfile():
    return render_template('upload.html')

@app.route('/download')  #
@cross_origin()
def download_file():
    p = './csv_file/result_output_data.csv'
    return send_file(p, as_attachment=True)



@app.route('/statistics',methods=['POST','GET'])  #
@cross_origin()
def stat_graph():
    return render_template('show_statistics.html')

if __name__ == "__main__":
    #clntApp = ClientApi()
    #host = '0.0.0.0'
    #port = 5000
    #app.run(debug=True)
    app.run(host='127.0.0.1', port=8001, debug=True)
    #httpd = simple_server.make_server(host, port, app)
    #print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()