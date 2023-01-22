import pandas as pd

import requests
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

import hyperopt
from hyperopt import fmin, tpe, hp
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def extract_file(): 
    # URL du fichier zip sur GitHub
    zip_file_url = "https://github.com/bastos12/xGoal_POO_optimizer/raw/master/data/events.zip"

    # Télécharger le fichier zip
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # extraire le fichier csv à l'intérieur du zip
    csv_file = [file for file in z.namelist() if file.endswith(".csv")][0]
    z.extract(csv_file)

    # Charger le fichier csv dans un dataframe pandas
    df = pd.read_csv(csv_file)
    return df

def prepa_data():
    df = extract_file()

    df.dropna(how='all',inplace=True)
    df.drop('player_in',axis=1,inplace=True)
    df.drop('player_out',axis=1,inplace=True)
    df.dropna(inplace=True)
    dum = df.select_dtypes(include='object')
    dumm =df.select_dtypes(exclude='object')
    X = dumm.drop('is_goal',axis=1)
    y = df['is_goal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
    
def regression_logistic(X_train, X_test, y_train, y_test):
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train,y_train)

    y_pred = logistic_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    print(classification_report(y_test,y_pred))

def model_knn(X_train, X_test, y_train, y_test):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train,y_train)
    y_pred = knn_model.predict(X_test)

    print(classification_report(y_test,y_pred))

def random_forest(X_train, X_test, y_train, y_test):
    random_model = RandomForestClassifier()
    random_model.fit(X_train, y_train)
    parameters = {'criterion' : ("gini", "entropy"),
              'n_estimators':[50,100,120,150,200]
              }
    random_opt_model = GridSearchCV(random_model, param_grid=parameters,cv=5,verbose=1)
    random_opt_model.fit(X_train,y_train)
    y_opt_pred = random_opt_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_opt_pred)
    print(classification_report(y_test,y_opt_pred))


def accueil(X_train, X_test, y_train, y_test):
    print("Quel model de classification souhaitez-vous choisir ?\n1- MODEL REGRESSION LOGISTIC\n2- MODEL KNN CLASSIFIER\n3- MODEL RANDOM FOREST \n4- Quit")
    choix = input("Merci d'entrez votre choix : ")
    if(choix == '1'):
        regression_logistic(X_train, X_test, y_train, y_test)
        return True
    elif(choix == '2'):
        model_knn(X_train, X_test, y_train, y_test)
        return True
    elif(choix == '3'):
        random_forest(X_train, X_test, y_train, y_test)
        return True
    elif(choix == '4'):
        print("Merci pour votre visite")
        return False
    else:
        print("Invalid choice")
        return True

    

if __name__ == '__main__':
    bool = True
    X_train, X_test, y_train, y_test = prepa_data()
    
    while bool:
        bool = accueil(X_train, X_test, y_train, y_test)

    


