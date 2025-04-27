import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("data.csv")
# df.head()
df = df.drop(['CholCheck','NoDocbcCost','DiffWalk','Sex','Education','Income'], axis='columns')
# df.head()
diseases = ['Diabetes_binary','Stroke','HeartDiseaseorAttack']
healthFact = ['HighBP','HighChol','BMI','HvyAlcoholConsump'
,'GenHlth','MentHlth','PhysHlth','Age' ]
# subset1 =df[diseases+healthFact]
# corelation = subset1.corr()
# diseaseCor = corelation.loc[diseases,healthFact]
# plt.figure(figsize=(12,6))
# sns.heatmap(diseaseCor,annot=True,cmap='coolwarm',fmt=".2f")
# plt.tight_layout()
# plt.show()

from sklearn.model_selection import train_test_split
y=df[ ['Diabetes_binary','Stroke','HeartDiseaseorAttack']]
x=df[['HighBP','HighChol','BMI','HvyAlcoholConsump'
,'GenHlth','MentHlth','PhysHlth','Age' ]]

# from imblearn.over_sampling import SMOTE
# smote = SMOTE()
# xResampled,yResampled = smote.fit_resample(x,y)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.7,random_state=42)


from sklearn.multioutput import MultiOutputClassifier


# from sklearn.ensemble import RandomForestClassifier
# baseModel= RandomForestClassifier(n_estimators=1000,random_state=42)


from xgboost import XGBClassifier
baseModel = XGBClassifier(
    scale_pos_weight=3.5,
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 3,
    random_state = 42,
    eval_metric = 'logloss'
)
model = MultiOutputClassifier(baseModel)
model.fit(xtrain,ytrain)
print(model.predict([[1,1,30,0,5,30,30,9]]))
print(model.score(xtest,ytest))