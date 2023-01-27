from flask import Flask, jsonify, request, render_template, url_for, session, redirect
import pandas as pd
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, StringField


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class DataLoader:
    """
        Cette méthode télécharge un fichier zip à partir d'une URL spécifique, 
        l'extrait et charge les données dans un DataFrame pandas.
    """

    @staticmethod
    def extract_file():
        zip_file_url = "https://github.com/bastos12/xGoal_POO_optimizer/raw/master/data/events.zip"
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_file = [file for file in z.namelist() if file.endswith(".csv")][0]
        z.extract(csv_file)
        df = pd.read_csv(csv_file)
        return df
    

    """
        Cette méthode enlève les lignes vides et supprime certaines colonnes inutiles du DataFrame.
    """
    @staticmethod
    def clean_data(df):
        df.dropna(how='all',inplace=True)
        df.drop('player_in',axis=1,inplace=True)
        df.drop('player_out',axis=1,inplace=True)
        df.dropna(inplace=True)
        return df
    
    """
        Cette méthode sépare les données en ensembles d'entraînement et de test.
    """
    @staticmethod
    def split_data(df):
        dum = df.select_dtypes(include='object')
        dumm =df.select_dtypes(exclude='object')
        X = dumm.drop('is_goal',axis=1)
        y = df['is_goal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
        return X_train, X_test, y_train, y_test

    
    """
        Cette méthode normalise les données en utilisant StandardScaler.
    """
    @staticmethod
    def scale_data(X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def prepa_data(self):
        df = self.extract_file()
        df = self.clean_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train, X_test = self.scale_data(X_train, X_test)
        return X_train, X_test, y_train, y_test

class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def regression_logistic(self):
        logistic_model = LogisticRegression()
        logistic_model.fit(self.X_train, self.y_train)
        #y_pred = logistic_model.predict(self.X_test)
        #ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred)
        return(classification_report(self.y_test, logistic_model.predict(self.X_test),output_dict=True))
        
    def model_knn(self):
        knn_model = KNeighborsClassifier()
        knn_model.fit(self.X_train, self.y_train)
        #y_pred = knn_model.predict(self.X_test)
        #print(classification_report(self.y_test, y_pred))
        return(classification_report(self.y_test, knn_model.predict(self.X_test),output_dict=True))
        
    def random_forest(self):
        random_model = RandomForestClassifier()
        random_model.fit(self.X_train, self.y_train)
        parameters = {'criterion' : ("gini", "entropy"), 'n_estimators':[50,100,120,150,200]}
        random_opt_model = GridSearchCV(random_model, param_grid=parameters, cv=5, verbose=1)
        random_opt_model.fit(self.X_train, self.y_train)
        y_opt_pred = random_opt_model.predict(self.X_test)
        ConfusionMatrixDisplay.from_predictions(self.y_test, y_opt_pred)
        print(classification_report(self.y_test, y_opt_pred))

class ModelForm(FlaskForm):
    model_choices = [('logistic', 'Logistic Regression'), ('knn', 'K-Nearest Neighbors'), ('random_forest', 'Random Forest')]
    model = RadioField('Model', choices=model_choices)

    submit = SubmitField('Submit')

@app.route('/select_model')
def select_model():
    form = ModelForm()
    return render_template('select_model.html', form=form)

@app.route('/regression_logistic', methods=['POST'])
def route_regression():
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.prepa_data()
    model = Model(X_train, X_test, y_train, y_test)
    report = model.regression_logistic()

    return report

@app.route('/knn')
def route__knn():
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.prepa_data()
    model = Model(X_train, X_test, y_train, y_test)
    report = model.model_knn()
    return report


@app.route("/", methods = ['GET', 'POST'])
def index():
    form = ModelForm()
    if form.validate_on_submit():
        session['model'] = form.model.data
        return redirect(url_for("prediction"))
    
    return render_template("home.html", form = form)

@app.route("/prediction")
def prediction():
    content = {}

    content['model'] = session['model']
    
    if(content['model'] == 'logistic'):
        results = route_regression()
    elif((content['model'] == 'knn')):
        results = route__knn()

    return render_template("prediction.html", results = results)
