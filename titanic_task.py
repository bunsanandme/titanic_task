import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# Загружаем данные и объединяем датафреймы
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
data = pd.concat([train_data, test_data], ignore_index=True, sort=False)

# Уникальные данные
print(data.nunique())

# Создание столбца "Семья" на основе родителей и детей
data['Family'] = data.Parch + data.SibSp

# Столбец "Путешествует один". Возникает тенденция на выживаемость одиночек
data['Is_Alone'] = data.Family == 0

# Столбец "Категория тарифа"
data['Fare_Category'] = pd.cut(data['Fare'],
                               bins=[0, 7.90, 14.45, 31.28, 120],
                               labels=['Low', 'Mid', 'High_Mid', 'High'])

# Заполнение пропусков в датафрейме
# В столбце "Место посадки" заменим пропущенные на моду выборки
data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)

data.Cabin = data.Cabin.fillna('NA')

# Замена пропущенных значений в столбце "Возраст"
# Выделение отдельного столбца "Обращение"
data['Salutation'] = data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

# Группировка и подстановка медианы
grp = data.groupby(['Sex', 'Pclass'])
grp.Age.apply(lambda x: x.fillna(x.median()))
data.Age.fillna(data.Age.median(), inplace=True)

# Перекодирование столбца "Пол" на 0 и 1
data['Sex'].replace('female', 0, inplace=True)
data['Sex'].replace('male', 1, inplace=True)

# Замена места отправления на числовые значения
data['Embarked'].replace('S', 0, inplace=True)
data['Embarked'].replace('C', 1, inplace=True)
data['Embarked'].replace('Q', 2, inplace=True)

# Удаление лишних столбцов для прогнозирования
data.drop(['Pclass', 'Fare', 'Cabin', 'Fare_Category', 'Name', 'Salutation', 'Ticket', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)

# Машинное обучение
# Данные для прогнозирования
X_to_be_predicted = data[data.Survived.isnull()]
X_to_be_predicted = X_to_be_predicted.drop(['Survived'], axis=1)

# Подготовка данных для обучения
train_data = data
train_data = train_data.dropna()
feature_train = train_data['Survived']
label_train = train_data.drop(['Survived'], axis=1)

# Прогнозирование данных при помощи случайного леса
clf = RandomForestClassifier(criterion='entropy',
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(label_train,
                                                    feature_train,
                                                    test_size=0.2)
clf.fit(x_train, np.ravel(y_train))
print("Точность случайного леса: " + repr(round(clf.score(x_test,
                                                          y_test) * 100,
                                                2)) + "%")
result_rf = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
print('Перекрестная оценка для случайного леса:', round(result_rf.mean()*100, 2))
y_pred = cross_val_predict(clf, x_train, y_train, cv=10)

# Построение матрицы ошибок для леса
sns.heatmap(confusion_matrix(y_train, y_pred), annot=True, fmt='3.0f', cmap="summer")
plt.title('Матрица ошибок случайного леса', y=1.05, size=15)

# Запись в файл "Тitanic Predictions.csv"
result = clf.predict(X_to_be_predicted)
submission = pd.DataFrame({'PassengerId':X_to_be_predicted.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)

filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False, sep=';')
print('Файл сохранен: ' + filename + "!")