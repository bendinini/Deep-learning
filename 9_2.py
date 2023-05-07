
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd

# чтение данных из CSV-файла
df = pd.read_csv('dataset.csv')

# разделение датасета на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.3, random_state=65, stratify=df.iloc[:,-1], train_size=0.7)

# стандартизация признаков тренировочных данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# применение стандартизации к тестовым данным
X_test = scaler.transform(X_test)

# обучение MLPClassifier и выполнение предсказания для тестовых объектов
clf = MLPClassifier(random_state=65, hidden_layer_sizes=(31, 10), activation='logistic', max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# выполнение предсказания для указанных тестовых объектов
test_data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
test_data = scaler.transform(test_data)
y_pred_test = clf.predict(test_data)

print("Результаты предсказания для тестовых данных:")
print(y_pred)
print("Результаты предсказания для указанных тестовых объектов:")
print(y_pred_test)
