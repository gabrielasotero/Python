import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
data = pd.read_csv('titanic_train.csv')

# Dados ausentes e tratamentos
sns.heatmap(data.isnull(), yticklabels=False)

#Preencher idades faltantes com a média das idades baseadas em suas respectivas classes
sns.boxplot(x='Pclass', y='Age', data=data)

def ages(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
data['Age'] = data[['Age', 'Pclass']].apply(ages,axis=1)

#Deletando a coluna Cabin e a linha Embarked que falta dado
data.drop('Cabin', axis=1, inplace=True)
data.dropna(inplace=True)

# Convertendo variáveis categóricas
sex = pd.get_dummies(data['Sex'], drop_first=True)
embark = pd.get_dummies(data['Embarked'], drop_first=True)

data.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
data = pd.concat([data,sex,embark], axis=1)

# Construindo um modelo de Regressão
#Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived', axis=1),
                                                     data['Survived'], test_size=0.30,
                                                     random_state=101)
#Predições
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test)

#Avaliação
print(classification_report(y_test,predictions))

