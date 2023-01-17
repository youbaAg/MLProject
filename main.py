from syslog import LOG_LOCAL0
from sqlite3 import Cursor


def accueil():
    print("Quel model de classification souhaitez-vous choisir ?\n1- MODEL REGRESSION LOGISTIC\n2- MODEL KNN CLASSIFIER\n3- MODEL RANDOM FOREST \n4- Quit")
    choix = input("Merci d'entrez votre choix : ")
    if(choix == '1'):
        print("MODEL REGRESSION LOGISTIC")
        return True
    elif(choix == '2'):
        print("MODEL KNN CLASSIFIER")
        return True
    elif(choix == '3'):
        print("MODEL RANDOM FOREST")
        return True
    elif(choix == '4'):
        print("Merci pour votre visite")
        return None
    else:
        print("Invalid choice")
        return None

    

if __name__ == '__main__':
    
    accueil()
    


