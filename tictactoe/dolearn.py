import tensorflow as tf
import numpy as np
import json
import os

# Включаем жадное выполнение для TensorFlow
tf.config.run_functions_eagerly(True)

# Функция для проверки победителя
def check_winner(board):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # горизонтальные
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # вертикальные
        [0, 4, 8], [2, 4, 6]              # диагональные
    ]
    for pos in win_positions:
        if board[pos[0]] == board[pos[1]] == board[pos[2]] != 0:
            return board[pos[0]]
    if 0 not in board:
        return 0  # Ничья
    return None  # Игра продолжается

# Пути к файлам
MODEL_PATH = '../models/tic_tac_toe_model.h5'
DATA_PATH = '../models/trainingtic_data.json'

# Загрузка данных для обучения
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
else:
    print("Ошибка: Файл с данными для обучения не найден!")
    exit()

# Подготовка данных
X = []
y = []

for board in data:
    X.append(board)
    result = check_winner(board)
    if result == 1:
        y.append([1] * 9)  # Победа игрока
    elif result == -1:
        y.append([-1] * 9)  # Победа нейросети
    else:
        y.append([0] * 9)  # Ничья

X = np.array(X)
y = np.array(y)

# Загрузка модели (если она существует)
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Модель загружена для дообучения.")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Модель создана с нуля.")

# Настройка параметров обучения
epochs = 500  # Увеличьте количество эпох, чтобы дать больше времени для обучения
batch_size = 32  # Размер пакета
learning_rate = 0.0001  # Низкая скорость обучения, чтобы не забыть старые данные

# Новый оптимизатор с уменьшенным learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Переобучение модели
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# Сохранение обновленной модели
model.save(MODEL_PATH)
print(f"Модель успешно дообучена и сохранена в {MODEL_PATH}.")
