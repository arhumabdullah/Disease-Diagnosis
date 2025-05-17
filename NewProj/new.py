import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("data.csv")

df = df.drop(['CholCheck','NoDocbcCost','DiffWalk','Sex','Education','Income'], axis='columns')

diseases = ['Diabetes_binary','Stroke','HeartDiseaseorAttack']
healthFact = ['HighBP','HighChol','BMI','HvyAlcoholConsump'
,'GenHlth','MentHlth','PhysHlth','Age' ]


y=df[ ['Diabetes_binary','Stroke','HeartDiseaseorAttack']]
x=df[['HighBP','HighChol','BMI','HvyAlcoholConsump'
,'GenHlth','MentHlth','PhysHlth','Age' ]]




from imblearn.over_sampling import SMOTE
xResampled = []
yResampled = []
lengths =[]
for i in range (y.shape[1]):
    smote = SMOTE(random_state=42)
    xi,yi = smote.fit_resample(x,y.iloc[:,i])
    indexResampled = yi.index
    xResampled.append(xi)
    yResampled.append(yi)
    lengths.append(len(xi))

minlength = min(lengths)

xFinal = xResampled[0].iloc[:minlength,:].reset_index(drop=True)
yFinal = pd.concat([yi.iloc[:minlength].reset_index(drop=True)for yi in yResampled],axis=1)
yFinal.columns = y.columns




from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(xFinal,yFinal,train_size=0.7,random_state=42)





from sklearn.multiclass import OneVsRestClassifier


from sklearn.ensemble import RandomForestClassifier
baseModel= RandomForestClassifier(n_estimators=100,random_state=42)

model = OneVsRestClassifier(baseModel)
model.fit(xtrain,ytrain)




from sklearn.metrics import accuracy_score, hamming_loss, f1_score, classification_report
y_pred = model.predict(xtest)
strict_acc = accuracy_score(ytest, y_pred)
hamming_acc = 1 - hamming_loss(ytest, y_pred)
f1 = f1_score(ytest, y_pred, average='samples')
print("\n🎯 Evaluation Metrics:")
print("-------------------------")
print("Strict Accuracy Score       :", round(strict_acc, 4))
print("Hamming Accuracy Score      :", round(hamming_acc, 4))
print("F1 Score (samples average)  :", round(f1, 4))
print("\n📊 Detailed Classification Report:")
print("-----------------------------------")
print(classification_report(ytest, y_pred, target_names=y.columns))
