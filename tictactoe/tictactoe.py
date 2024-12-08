import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
import sys
import os
import json

# Путь к файлу модели и данным
MODEL_PATH = '../models/tic_tac_toe_model.h5'
DATA_PATH = '../models/trainingtic_data.json'

# Загрузка обученной модели
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Ошибка: Модель не найдена!")
    sys.exit(1)

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
        self.game_data = []  # Для хранения данных о сыгранных партиях

    def initUI(self):
        self.setWindowTitle('Крестики-нолики')
        self.setGeometry(100, 100, 400, 450)

        self.board = [0] * 9  # Игровая доска

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

        # Кнопка для новой игры
        new_game_button = QPushButton('Новая игра')
        new_game_button.clicked.connect(self.reset_game)
        layout.addWidget(new_game_button)

        # Кнопка для игры вдвоём
        self.two_player_mode = False
        toggle_mode_button = QPushButton('Режим: Игра против нейросети')
        toggle_mode_button.clicked.connect(self.toggle_mode)
        layout.addWidget(toggle_mode_button)

        self.setLayout(layout)
        self.show()

    def make_move(self, index):
        if self.board[index] == 0:
            self.board[index] = 1 if not self.two_player_mode or sum(np.abs(self.board)) % 2 == 0 else -1
            self.button_grid[index].setText('X' if self.board[index] == 1 else 'O')
            winner = check_winner(self.board)
            if winner is not None:
                self.display_winner(winner)
                return

            # Ход нейросети (если режим против ИИ)
            if not self.two_player_mode:
                ai_move(self.board, model, self.button_grid)
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

        # Сохранение текущей партии
        self.save_game_data()

    def reset_game(self):
        self.board = [0] * 9
        self.status_label.setText('Ваш ход: X')
        for button in self.button_grid:
            button.setText('')

    def toggle_mode(self):
        self.two_player_mode = not self.two_player_mode
        mode_text = 'Режим: Игра вдвоём' if self.two_player_mode else 'Режим: Игра против нейросети'
        self.status_label.setText(mode_text)

    def save_game_data(self):
        if not hasattr(self, 'game_data'):
            self.game_data = []
        self.game_data.append(self.board.copy())
        with open(DATA_PATH, 'w') as f:
            json.dump(self.game_data, f)

# Запуск игры
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TicTacToeApp()
    sys.exit(app.exec_())
