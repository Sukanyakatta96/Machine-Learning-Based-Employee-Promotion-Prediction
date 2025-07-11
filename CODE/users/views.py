from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Employee_promotion_prediction

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'train.csv'
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the data
path = os.path.join(settings.MEDIA_ROOT, 'train.csv')
df = pd.read_csv(path)
df.fillna(0, inplace=True)

le=LabelEncoder()
df['department']=le.fit_transform(df['department'])
df['region']=le.fit_transform(df['region'])
df['gender']=le.fit_transform(df['gender'])
df['recruitment_channel']=le.fit_transform(df['recruitment_channel'])
df['no_of_trainings']=le.fit_transform(df['no_of_trainings'])
df['age']=le.fit_transform(df['age'])
df['previous_year_rating']=le.fit_transform(df['previous_year_rating'])
df['length_of_service']=le.fit_transform(df['length_of_service'])
df['awards_won']=le.fit_transform(df['awards_won'])
df['avg_training_score']=le.fit_transform(df['avg_training_score'])
df['is_promoted']=le.fit_transform(df['is_promoted'])

X = df[['department', 'region', 'gender', 'recruitment_channel', 
         'no_of_trainings', 'age', 'previous_year_rating', 
         'length_of_service', 'awards_won', 'avg_training_score']]
y = df['is_promoted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_resampled, y_resampled)
y_pred = rfc.predict(X_test)

def Training(request):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Accuracy: {accuracy:.2f}')
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    y_train_pred = rfc.predict(X_train)
    y_test_pred = rfc.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Train Accuracy: {train_acc * 100: .2f}%")
    print(f"Test Accuracy: {test_acc * 100: .2f}%")
    y_pred_proba = rfc.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    context = {
       'train_acc': train_acc,
       'test_acc': test_acc,
    }
    return render(request, 'users/ml.html', context)

def Prediction(request):
    if request.method == "POST":
        dept = request.POST.get('department')
        re = request.POST.get('region')
        ge = request.POST.get('gender')
        rc = request.POST.get('recruitment_channel')
        nt = request.POST.get('no_of_trainings')
        ag = request.POST.get('age')
        pyr = request.POST.get('previous_year_rating')
        los = request.POST.get('length_of_service')
        ao = request.POST.get('awards_won')
        ats = request.POST.get('avg_training_score')
        nt = int(nt) if nt else 0
        ag = int(ag) if ag else 0
        pyr = float(pyr) if pyr else 0.0
        los = int(los) if los else 0
        ats = float(ats) if ats else 0.0
        ao = 1 if ao == 'Yes' else 0
        gender_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
        ge = gender_mapping.get(ge, 2)  # Default to 'Other' (2) if not Male or Female
        region_mapping = {'region_1': 0, 'region_2': 1, 'region_3': 2}          
        department_mapping = {'HR': 0, 'Sales': 1, 'Finance': 2, 'Engineering': 3}
        recruitment_channel_mapping = {'sourcing': 0, 'referred': 1, 'other': 2}
        dept = department_mapping.get(dept, 0)
        re = region_mapping.get(re, 0)  
        rc = recruitment_channel_mapping.get(rc, 0)
        input_df = pd.DataFrame({
            'department': [dept],
            'region': [re],
            'gender': [ge],
            'recruitment_channel': [rc],
            'no_of_trainings': [nt],
            'age': [ag],
            'previous_year_rating': [pyr],
            'length_of_service': [los],
            'awards_won': [ao],
            'avg_training_score': [ats]
        })
        print("Input DataFrame before encoding:", input_df)
        op = rfc.predict(input_df)
        prediction_label = "Promoted" if int(op[0]) == 1 else "Not Promoted"
        print("Prediction label:", prediction_label)
        context = {
            'prediction': prediction_label
        }

        return render(request, 'users/predict_form.html', context)

    return render(request, 'users/predict_form.html')
