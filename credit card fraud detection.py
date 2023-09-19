import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Загрузка данных
data = pd.read_csv('/Users/dmitriy/Downloads/creditcard_2023.csv')

# Посмотрим первые строки датасета
print(data.head())

# Выводим общую статистику по датасету
print(data.describe())

# Выводим количество обычных и мошеннических транзакций
fraud_count = data[data['Class'] == 1].shape[0]
normal_count = data[data['Class'] == 0].shape[0]
print(f"Количество обычных транзакций: {normal_count}")
print(f"Количество мошеннических транзакций: {fraud_count}")

sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Распределение суммы транзакций по классам')
plt.show()


# Отделяем признаки от целевой переменной
X = data.drop(['Class', 'id'], axis=1)  # Удаляем столбцы Class и id
y = data['Class']

# Разделяем данные на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# Создаем объект масштабирования
scaler = StandardScaler()

# Обучаем масштабирование на обучающих данных и применяем его к обучающей и тестовой выборке
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Создание и обучение модели
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Предсказание на тестовой выборке
y_pred = logistic_model.predict(X_test_scaled)

# Оценка модели
print("Оценка производительности модели:")
print(classification_report(y_test, y_pred))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print(f"Точность модели: {accuracy_score(y_test, y_pred):.4f}")



# Создание и обучение модели
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Для случайного леса масштабирование данных не требуется

# Предсказание на тестовой выборке
y_pred_rf = rf_model.predict(X_test)

# Оценка модели
print("Оценка производительности модели случайного леса:")
print(classification_report(y_test, y_pred_rf))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_rf))
print(f"Точность модели: {accuracy_score(y_test, y_pred_rf):.4f}")
