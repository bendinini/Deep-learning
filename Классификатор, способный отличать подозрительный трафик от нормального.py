
# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Загружаем датасет
df = pd.read_csv('360T.csv')

# Выделяем предикторы и отклики
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Разделяем данные на тренировочный и тестовый наборы с параметрами test_size = 0.3, random_state = 65
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=65, stratify=y)

# Стандартизируем признаки тренировочных данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Применяем полученное преобразование для тестовых данных
X_test = scaler.transform(X_test)

# Обучаем классификатор MLPClassifier при random_state = 65, hidden_layer_sizes = (31, 10), activation = 'logistic', max_iter = 1000 на обучающей выборке
clf = MLPClassifier(random_state=65, hidden_layer_sizes=(31, 10), activation='logistic', max_iter=1000)
clf.fit(X_train, y_train)

# Производим оценку полученной модели на тестовой выборке
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загружаем датасет
df = pd.read_csv('360T.csv')

# Выделяем предикторы и отклики
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Разделяем данные на тренировочный и тестовый наборы с параметрами test_size = 0.3, random_state = 65
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=65, stratify=y)

# Стандартизируем признаки тренировочных данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Применяем полученное преобразование для тестовых данных
X_test = scaler.transform(X_test)

# Обучаем классификатор MLPClassifier при random_state = 65, hidden_layer_sizes = (31, 10), activation = 'logistic', max_iter = 1000 на обучающей выборке
clf = MLPClassifier(random_state=65, hidden_layer_sizes=(31, 10), activation='logistic', max_iter=1000)
clf.fit(X_train, y_train)

# Производим оценку полученной модели на тестовой выборке
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f_score = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision (macro avg): {precision:.2f}')
print(f'Recall (macro avg): {recall:.2f}')
print(f'F-score (macro avg): {f_score:.2f}')
