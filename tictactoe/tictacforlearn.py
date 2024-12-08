import numpy as np
import tensorflow as tf
import random
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
import sys
import os

# Путь к файлам модели и данных
MODEL_PATH = '../models/tic_tac_toe_model.h5'
DATA_PATH = '../models/training_data_for_tictac.pkl'

# Загрузка обученной модели и данных
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Ошибка: Модель не найдена!")
    sys.exit(1)

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        X, y = pickle.load(f)
        # Конвертируем обратно в списки
        X = X.tolist() if isinstance(X, np.ndarray) else X
        y = y.tolist() if isinstance(y, np.ndarray) else y
else:
    X, y = [], []

# Функция для проверки победителя
def check_winner(board):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # горизонтальные
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # вертикальные
        [0, 4, 8], [2, 4, 6]  # диагональные
    ]
    for pos in win_positions:
        if board[pos[0]] == board[pos[1]] == board[pos[2]] != 0:
            return board[pos[0]]
    if 0 not in board:
        return 0  # Ничья
    return None  # Игра продолжается

# Функция для хода нейросети
def ai_move(board, model, button_grid):
    board_input = np.array([board])
    prediction = model.predict(board_input, verbose=0)
    best_move = np.argmax(prediction)
    while board[best_move] != 0:
        prediction[0][best_move] = -1
        best_move = np.argmax(prediction)
    board[best_move] = -1
    button_grid[best_move].setText('O')

# Основной класс игры в PyQt
class TicTacToeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Крестики-нолики')
        self.setGeometry(100, 100, 400, 400)

        self.board = [0] * 9  # Игровая доска
        self.game_data = []  # Данные для обучения

        # Инициализация интерфейса
        layout = QVBoxLayout()

        self.status_label = QLabel('Ваш ход: X')
        layout.addWidget(self.status_label)

        grid_layout = QGridLayout()
        self.button_grid = []

        for i in range(9):
            button = QPushButton('')
            button.setFixedSize(100, 100)
            button.clicked.connect(lambda _, i=i: self.make_move(i))
            self.button_grid.append(button)
            row = i // 3
            col = i % 3
            grid_layout.addWidget(button, row, col)

        layout.addLayout(grid_layout)
        self.setLayout(layout)

        self.show()

    def make_move(self, index):
        if self.board[index] == 0:
            self.board[index] = 1
            self.button_grid[index].setText('X')
            self.game_data.append((self.board.copy(), 1))

            winner = check_winner(self.board)
            if winner is not None:
                self.display_winner(winner)
                return

            # Ход нейросети
            ai_move(self.board, model, self.button_grid)
            self.game_data.append((self.board.copy(), -1))

            winner = check_winner(self.board)
            if winner is not None:
                self.display_winner(winner)

    def display_winner(self, winner):
        if winner == 1:
            self.status_label.setText('Поздравляем! Вы победили!')
        elif winner == -1:
            self.status_label.setText('Нейросеть победила!')
        else:
            self.status_label.setText('Ничья!')

        self.do_training()
        self.reset_game()

    def do_training(self):
        global model, X, y

        # Сохранение данных текущей игры
        for state, label in self.game_data:
            X.append(state)
            y.append([label] * 9)

        # Подготовка данных для обучения
        X_data = np.array(X)
        y_data = np.array(y)

        # Перекомпилируем модель перед дообучением
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Обучаем модель на новых данных
        model.fit(X_data, y_data, epochs=5, batch_size=32, verbose=1)

        # Сохраняем обновлённую модель и данные
        model.save(MODEL_PATH)
        with open(DATA_PATH, 'wb') as f:
            pickle.dump((X, y), f)

        print("Модель успешно дообучена и сохранена!")

    def reset_game(self):
        self.board = [0] * 9
        self.game_data = []
        self.status_label.setText('Ваш ход: X')
        for button in self.button_grid:
            button.setText('')

# Запуск игры
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TicTacToeApp()
    sys.exit(app.exec_())
