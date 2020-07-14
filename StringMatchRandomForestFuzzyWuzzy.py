#!/usr/bin/env python
# coding: utf-8

# # Importing the library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz,process


# # Importing the excel

# In[2]:


df= pd.read_excel('MatchPairs.xlsx', sheet_name='Sheet2')
df = df.replace(np.nan, '', regex=True)
df.head(2)


# # Levenshtein Distance FUnction

# In[3]:


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
  
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y
        
    

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


# In[4]:


print(levenshtein("oldKundan","Kundanold"))


# In[5]:


def sorted_levenshtein(sequence1, sequence2):
    product1 = ''.join(sorted(sequence1))
    product2 = ''.join(sorted(sequence2))
    distance = levenshtein(product1, product2)
    return distance

def levenshtein_rate(product1, product2):
    distance = levenshtein(product1, product2)
    max_len = max(len(product1), len(product2))
    return 1 - (distance / max_len)

def sorted_levenshtein_rate(seq1, seq2):
    product1 = ''.join(sorted(seq1))
    product2 = ''.join(sorted(seq2))
    distance = levenshtein(product1, product2)
    max_len = max(len(product1), len(product2))
    return 1-(distance/max_len)


# In[6]:


matchScore = pd.DataFrame([])


# In[7]:


for index,row in df.iterrows():
    matchScore = matchScore.append(pd.DataFrame({
             'partyid1':row['PARTY_ID_1'],
             'partyid2':row['PARTY_ID_2'],
             #'full_name1':row['FULL_NAME_1'],
             #'full_name2':row['FULL_NAME_2'],
             #'nameScore':fuzz.token_set_ratio(str(row['FULL_NAME_1']).upper(),str(row['FULL_NAME_2']).upper())/100 ,
             'first_name1':row['FIRSTNAME_1'],
             'first_name2':row['FIRSTNAME_2'],
             'firstnameScore':fuzz.token_set_ratio(str(row['FIRSTNAME_1']).upper(),str(row['FIRSTNAME_2']).upper())/100 ,
            #'nameScore':sorted_levenshtein_rate(str(row['FULL_NAME_1']).upper(),str(row['FULL_NAME_2']).upper()) ,
             'last_name1':row['LASTNAME_1'],
             'last_name2':row['LASTNAME_2'],
             'lastnameScore':fuzz.token_set_ratio(str(row['LASTNAME_1']).upper(),str(row['LASTNAME_2']).upper())/100 ,
             'mobile1':row['MOBILE_1'],
             'mobile2':row['MOBILE_2'],
             'mobileScore': fuzz.ratio(str(row['MOBILE_1']),str(row['MOBILE_2']))/100, 
             'private1':row['PRIVATE_1'],
             'private2':row['PRIVATE_2'],
             'privateScore': fuzz.ratio(str(row['PRIVATE_1']),str(row['PRIVATE_2']))/100, 
             'work1':row['WORK_1'],
             'work2':row['WORK_2'],
             'workScore': fuzz.ratio(str(row['WORK_1']),str(row['WORK_2']))/100, 
             'email1':row['ELECTRONIC_ADDRESS_1'],
             'email2':row['ELECTRONIC_ADDRESS_2'],
             'emailScore': fuzz.partial_ratio(str(row['ELECTRONIC_ADDRESS_1']).upper(),str(row['ELECTRONIC_ADDRESS_2']).upper())/100, 
             'addressLine1':row['ST_ADDRESS_LINE_1'],
             'addressLine2':row['ST_ADDRESS_LINE_2'],
             
             'addressLineScore': fuzz.token_sort_ratio(str(row['ST_ADDRESS_LINE_1']).upper(),str(row['ST_ADDRESS_LINE_2']).upper())/100,
             'city1':row['ST_CITY_1'],
             'city2':row['ST_CITY_2'],
             'cityScore': fuzz.partial_ratio(str(row['ST_CITY_1']).upper(),str(row['ST_CITY_2']).upper())/100,
             'post1':row['ST_POSTAL_CODE_1'],
             'post2':row['ST_POSTAL_CODE_2'],
             'postScore':fuzz.ratio(str(row['ST_POSTAL_CODE_1']).upper(),str(row['ST_POSTAL_CODE_2']).upper())/100, 
              'Match':row['Match']
            },index=[0]), ignore_index=True)


# In[8]:


matchScore.to_excel('MatchResults7.xlsx')


# In[9]:


df_reg=matchScore[['firstnameScore','lastnameScore','mobileScore','privateScore','workScore','emailScore','addressLineScore','cityScore','postScore','Match']]
#df_regShow=matchScore
#df_reg=matchScore[['nameScore','mobileScore','privateScore','workScore','emailScore','addressLineScore','cityScore','postScore','Match']]


# In[10]:


x = df_reg.iloc[:, :-1].values
y = df_reg.iloc[:, -1].values
y=y.astype('int')


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[13]:


y_pred = classifier.predict(X_test)


# In[14]:


#print(np.concatenate((X_test,y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#print(np.concatenate([y_pred,y_test], axis=0)
final = pd.DataFrame(np.concatenate((X_test,y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
print(final)
final.to_excel('MatchResults6.xlsx')


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[16]:


#print(fuzz.partial_ratio('OLAV','SILJE'))
#print(fuzz.token_set_ratio('OLAV','SILJE'))
print(fuzz.token_sort_ratio("nan","STØLHAUGVEGEN 3"))
#print(fuzz.ratio('OLAV','SILJE'))
print(sorted_levenshtein_rate('','SILJE'))


# In[17]:


print(classifier.predict([[fuzz.token_set_ratio('IGOR'.upper(),'IGOR'.upper())/100 ,
                           fuzz.token_set_ratio('KICA'.upper(),'KICA'.upper())/100 ,
                           fuzz.ratio('96824370','96824370')/100, 
                           fuzz.ratio('','')/100, 
                           fuzz.ratio('96824370','')/100, 
                           fuzz.partial_ratio('IGOR.KICA@GMAIL.COM'.upper(),'IGOR.KICA@GMAIL.COM'.upper())/100, 
                           fuzz.token_sort_ratio('LØKKAVEGEN 11 C'.upper(),'RISKOLLVEIEN 27'.upper())/100,
                           fuzz.partial_ratio('JESSHEIM'.upper(),'ØSTERÅS'.upper())/100,
                           fuzz.ratio('2052'.upper(),'1332'.upper())/100]]))


# In[18]:


testDf= pd.read_excel('TestDataSet1.xlsx', sheet_name='Sheet3')
testDf = testDf.replace(np.nan, '', regex=True)

testPrediction = pd.DataFrame([])
testPrediction["Match"]= ""
testPrediction.head(2)


# In[19]:


for index,row in testDf.iterrows():
  
     #print(row['PARTY_ID_1'])
     testPrediction = testPrediction.append(pd.DataFrame({
             
             'partyid1':row['PARTY_ID_1'],
             'partyid2':row['PARTY_ID_2'],
             #'full_name1':row['FULL_NAME_1'],
             #'full_name2':row['FULL_NAME_2'],
             #'nameScore':fuzz.token_set_ratio(str(row['FULL_NAME_1']).upper(),str(row['FULL_NAME_2']).upper())/100 ,
             'first_name1':row['FIRSTNAME_1'],
             'first_name2':row['FIRSTNAME_2'],
             'firstnameScore':fuzz.token_set_ratio(str(row['FIRSTNAME_1']).upper(),str(row['FIRSTNAME_2']).upper())/100 ,
            #'nameScore':sorted_levenshtein_rate(str(row['FULL_NAME_1']).upper(),str(row['FULL_NAME_2']).upper()) ,
             'last_name1':row['LASTNAME_1'],
             'last_name2':row['LASTNAME_2'],
             'lastnameScore':fuzz.token_set_ratio(str(row['LASTNAME_1']).upper(),str(row['LASTNAME_2']).upper())/100 ,
             'mobile1':row['MOBILE_1'],
             'mobile2':row['MOBILE_2'],
             'mobileScore': fuzz.ratio(str(row['MOBILE_1']),str(row['MOBILE_2']))/100, 
             'private1':row['PRIVATE_1'],
             'private2':row['PRIVATE_2'],
             'privateScore': fuzz.ratio(str(row['PRIVATE_1']),str(row['PRIVATE_2']))/100, 
             'work1':row['WORK_1'],
             'work2':row['WORK_2'],
             'workScore': fuzz.ratio(str(row['WORK_1']),str(row['WORK_2']))/100, 
             'email1':row['ELECTRONIC_ADDRESS_1'],
             'email2':row['ELECTRONIC_ADDRESS_2'],
             'emailScore': fuzz.partial_ratio(str(row['ELECTRONIC_ADDRESS_1']).upper(),str(row['ELECTRONIC_ADDRESS_2']).upper())/100, 
             'addressLine1':row['ST_ADDRESS_LINE_1'],
             'addressLine2':row['ST_ADDRESS_LINE_2'],
             
             'addressLineScore': fuzz.token_sort_ratio(str(row['ST_ADDRESS_LINE_1']).upper(),str(row['ST_ADDRESS_LINE_2']).upper())/100,
             'city1':row['ST_CITY_1'],
             'city2':row['ST_CITY_2'],
             'cityScore': fuzz.partial_ratio(str(row['ST_CITY_1']).upper(),str(row['ST_CITY_2']).upper())/100,
             'post1':row['ST_POSTAL_CODE_1'],
             'post2':row['ST_POSTAL_CODE_2'],
             'postScore':fuzz.ratio(str(row['ST_POSTAL_CODE_1']).upper(),str(row['ST_POSTAL_CODE_2']).upper())/100, 
             'Match':classifier.predict([[
                           fuzz.token_set_ratio(str(row['FIRSTNAME_1']).upper(),str(row['FIRSTNAME_2']).upper())/100 ,
                           fuzz.token_set_ratio(str(row['LASTNAME_1']).upper(),str(row['LASTNAME_2']).upper())/100 ,
                           fuzz.ratio(str(row['MOBILE_1']),str(row['MOBILE_2']))/100,  
                           fuzz.ratio(str(row['PRIVATE_1']),str(row['PRIVATE_2']))/100, 
                           fuzz.ratio(str(row['WORK_1']),str(row['WORK_2']))/100, 
                           fuzz.partial_ratio(str(row['ELECTRONIC_ADDRESS_1']).upper(),str(row['ELECTRONIC_ADDRESS_2']).upper())/100, 
                           fuzz.token_sort_ratio(str(row['ST_ADDRESS_LINE_1']).upper(),str(row['ST_ADDRESS_LINE_2']).upper())/100,
                           fuzz.partial_ratio(str(row['ST_CITY_1']).upper(),str(row['ST_CITY_2']).upper())/100,
                           fuzz.ratio(str(row['ST_POSTAL_CODE_1']).upper(),str(row['ST_POSTAL_CODE_2']).upper())/100
                              ]])
            },index=[0]), ignore_index=True)
    


# In[20]:


testPrediction.to_excel('TestSetPrediction.xlsx')


# In[ ]:




