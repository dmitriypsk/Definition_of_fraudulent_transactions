import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE


def оценить_модель(model, X_test, y_test, название_модели="Модель"):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Оценка {название_модели}:")
    print(classification_report(y_test, y_pred))
    print("Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}\n")


# Загрузка данных
data = pd.read_csv('/Users/dmitriy/Downloads/creditcard_2023.csv')

print(data.head())
print(data.describe())

fraud_count = data[data['Class'] == 1].shape[0]
normal_count = data[data['Class'] == 0].shape[0]
print(f"Количество обычных транзакций: {normal_count}")
print(f"Количество мошеннических транзакций: {fraud_count}")

sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Распределение сумм транзакций по классам')
plt.show()

X = data.drop(['Class', 'id'], axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Логистическая регрессия
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_resampled_scaled, y_train_resampled)
оценить_модель(logistic_model, X_test_scaled, y_test, "Логистическая регрессия")

# Случайный лес
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, y_train_resampled)
оценить_модель(rf_model, X_test, y_test, "Случайный лес")

# Кросс-валидация с логистической регрессией
model = LogisticRegression()
scores = cross_val_score(model, X_train_resampled_scaled, y_train_resampled, cv=5)
print("Оценки кросс-валидации:", scores)
print("Среднее значение оценок кросс-валидации:", scores.mean())

train_sizes, train_scores, valid_scores = learning_curve(model, X_train_resampled_scaled, y_train_resampled, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Обучающая выборка')
plt.plot(train_sizes, valid_scores.mean(axis=1), label='Тестовая выборка')
plt.legend()
plt.title('Кривая обучения')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Оценка')
plt.show()

# Важность признаков в случайном лесе
rf_model_for_importance = RandomForestClassifier()
rf_model_for_importance.fit(X_train_resampled_scaled, y_train_resampled)
feature_importance = rf_model_for_importance.feature_importances_
sorted_idx = feature_importance.argsort()

plt.figure(figsize=(10, 15))
plt.barh(range(X_train_resampled_scaled.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X_train_resampled_scaled.shape[1]), X.columns[sorted_idx])
plt.xlabel('Важность')
plt.title('Важность признаков в случайном лесе')
plt.show()
