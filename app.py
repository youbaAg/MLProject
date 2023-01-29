from flask import Flask, jsonify, request, render_template, url_for, session, redirect
import pandas as pd
import requests
import pickle
import zipfile
import io
import os
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
from wtforms import RadioField, SubmitField,BooleanField


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class ModelForm(FlaskForm):
    model_choices = [('logistic', 'Logistic Regression'), ('knn', 'K-Nearest Neighbors'), ('random_forest', 'Random Forest')]
    model = RadioField('Model', choices=model_choices)
    optimize = BooleanField('choice')
    submit = SubmitField('Submit')


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
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
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
        
    def train_save_model_regression(self):
        if os.path.exists('logistic_regression_model.pkl'):
            return
        else:
            # Initialiser le modèle de régression logistique
            logistic_model = LogisticRegression()
            # Appliquer le modèle aux données d'entraînement
            logistic_model.fit(self.X_train, self.y_train)
            # Ouvrir un fichier pour enregistrer le modèle
            with open('logistic_regression_model.pkl', 'wb') as file:
                # Enregistrer le modèle dans le fichier
                pickle.dump(logistic_model, file)
    
    def train_save_model_knn(self):
        if os.path.exists('knn_model.pkl'):
            return
        else:
            # Initialiser le modèle knn
            knn_model = KNeighborsClassifier()
            # Appliquer le modèle aux données d'entraînement
            knn_model.fit(self.X_train, self.y_train)
            # Ouvrir un fichier pour enregistrer le modèle
            with open('knn_model.pkl', 'wb') as file:
                # Enregistrer le modèle dans le fichier
                pickle.dump(knn_model, file)
    
    def train_save_model_random_forest(self):
        if os.path.exists('random_model.pkl'):
            return
        else:
            # Initialiser le modèle random forest
            random_model = RandomForestClassifier()
            # Appliquer le modèle aux données d'entraînement
            random_model.fit(self.X_train, self.y_train)
            # Ouvrir un fichier pour enregistrer le modèle
            with open('random_model.pkl', 'wb') as file:
                # Enregistrer le modèle dans le fichier
                pickle.dump(random_model, file)

    def load_model_regression(self):
        # Ouvrir le fichier où le modèle est enregistré
        with open('logistic_regression_model.pkl', 'rb') as file:
            # Charger le modèle à partir du fichier
            model_loaded = pickle.load(file)
        return model_loaded

    def load_model_knn(self):
        # Ouvrir le fichier où le modèle est enregistré
        with open('knn_model.pkl', 'rb') as file:
            # Charger le modèle à partir du fichier
            model_loaded = pickle.load(file)
        return model_loaded
    
    def load_model_random_forest(self):
        # Ouvrir le fichier où le modèle est enregistré
        with open('random_model.pkl', 'rb') as file:
            # Charger le modèle à partir du fichier
            model_loaded = pickle.load(file)
        return model_loaded
    
    def display_results(self,model_loaded):
        # Afficher les résultats de classification en utilisant les données de test
        return classification_report(self.y_test, model_loaded.predict(self.X_test),output_dict=True)



@app.route("/", methods = ['GET', 'POST'])
def index():
    form = ModelForm() 
    if form.validate_on_submit():
        session['model'] = form.model.data
        session['optimize'] = form.optimize.data

        return redirect(url_for("prediction")) 
    return render_template("home.html", form = form)

@app.route("/prediction")
def prediction():
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.prepa_data()
    model = Model(X_train, X_test, y_train, y_test)

    content = {}

    content['model'] = session['model']

    if(content['model'] == 'logistic'):
        model.train_save_model_regression()
        result_logistic = model.display_results(model.load_model_regression())
        results = result_logistic
    elif((content['model'] == 'knn')):
        model.train_save_model_knn()
        result_knn = model.display_results(model.load_model_knn())
        results = result_knn
    elif((content['model'] == 'random_forest')):
        model.train_save_model_random_forest()
        result_random = model.display_results(model.load_model_random_forest())
        results = result_random

    return render_template("prediction.html", results = results)
