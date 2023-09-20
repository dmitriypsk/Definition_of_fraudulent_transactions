# **Анализ мошеннических транзакций с использованием машинного обучения**
**Обзор**
Код приведенный выше анализирует набор данных о кредитных картах с целью определения мошеннических транзакций. В основном используются инструменты машинного обучения из библиотек sklearn и imblearn.

**Описание процесса**
**Подготовка библиотек:** Импортируются все необходимые библиотеки.

**Оценка модели:** Функция оценить_модель предназначена для оценки производительности модели машинного обучения. Она выводит различные метрики качества.

**Загрузка данных:** Данные о транзакциях кредитных карт загружаются из CSV файла.

**Анализ данных:** Краткий анализ данных с распределением по классам и визуализацией.

**Предварительная обработка данных:**

  Исключение ненужных столбцов.
  Разделение данных на обучающую и тестовую выборки.
  Применение SMOTE для устранения дисбаланса классов в данных.
  Масштабирование признаков.

**Обучение моделей:**

  **Логистическая регрессия:** Данные обучаются с помощью логистической регрессии и затем оцениваются.
  **Случайный лес:** Аналогично логистической регрессии, но используется модель случайного леса.
  
**Кросс-валидация:** Производится кросс-валидация с логистической регрессией.

**Кривая обучения:** Визуализация кривой обучения, которая показывает, как качество модели зависит от количества данных, используемых для обучения.

**Важность признаков:** Отображение важности различных признаков при использовании модели случайного леса.

**Заключение**
Этот код предоставляет всеобъемлющий подход к анализу и классификации мошеннических транзакций на основе данных о кредитных картах. Применяются различные техники машинного обучения для построения и оценки моделей.
