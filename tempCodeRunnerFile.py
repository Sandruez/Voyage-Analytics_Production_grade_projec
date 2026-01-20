import pandas as pd
from voyage_analytics.entity.local_estimator import Local_Estimator_Class

obj=Local_Estimator_Class()

# # ['from', 'to', 'flightType', 'time', 'distance', 'agency', 'day']
df=pd.DataFrame({'from':['Brasilia (DF)'], 'to':['Recife (PE)'], 'flightType':['economic'], 'time':[1.40], 'distance':[242.21], 'agency':['Rainbow'], 'day':['Monday']})
preds=obj.classification_predict_func(df)

print(preds,'\n')
print(type(preds))


# travelCode,userCode,from,to,flightType,price,time,distance,agency,date
# 25970,250,Brasilia (DF),Recife (PE),economic,421.2,0.63,242.21,Rainbow,03/07/2020
