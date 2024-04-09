import numpy as np
import random as rd


######


class TicTacToeAI:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y):
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        output = self.forward(X)
        return output

    def convert_board_to_input(self, board):
        X = []
        for i in range(len(board)):
            if board[i] == 'X':
                X.append(1)
            elif board[i] == 'O':
                X.append(-1)
            else:
                X.append(0)
        return np.array(X).reshape(1, -1)


######


empty = ' '
board = [empty for _ in range(9)]


######


def draw_board(board):
    print('-------------')
    print('|', board[0], '|', board[1], '|', board[2], '|')
    print('-------------')
    print('|', board[3], '|', board[4], '|', board[5], '|')
    print('-------------')
    print('|', board[6], '|', board[7], '|', board[8], '|')
    print('-------------')


######


def check_win(player, board):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 가로
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 세로
        [0, 4, 8], [2, 4, 6]  # 대각선
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

def play_game():

    train_itr = int(input("인공지능을 몇회 학습시키겠습니까?: "))
    print()
    print(f'{train_itr}회 학습을 시작합니다.')

    model = TicTacToeAI(input_size=9, hidden_size=16, output_size=9, learning_rate=0.1)

    iteration = 0

    while True:
        board = [empty for _ in range(9)]
        current_player = 'X'
        p_error = 0
        while True:
            if p_error != 1:
                print()
                draw_board(board)
                print("현재 차례: ", current_player)
            p_error = 0
            if current_player == 'X':
                if iteration >= train_itr:
                    position = input("원하는 위치를 선택하세요 (0-8): ")
                else:
                    position = rd.randint(0,8)
                if not str(position).isdigit():
                    print("잘못된 입력입니다. 숫자를 입력하세요.")
                    continue
                position = int(position)
                if position < 0 or position > 8:
                    print("잘못된 입력입니다. 0에서 8 사이의 숫자를 입력하세요.")
                    continue
                if board[position] != empty:
                    if iteration <= train_itr:
                        p_error = 1
                    else:
                        print("이미 선택된 위치입니다. 다른 위치를 선택하세요.")
                    continue
            else:
                X = model.convert_board_to_input(board)
                output = model.predict(X)
                print(output)

                position = np.argmax(output)
                while board[position] != empty:
                    output[0, position] = -np.inf  # 이미 선택된 위치는 선택되지 않도록 확률을 음의 무한대로 설정
                    position = np.argmax(output)
                    print(output)

            board[position] = current_player
            print()

            if check_win(current_player, board):
                draw_board(board)
                print()
                print("게임 종료! 승자는", current_player, "입니다!")
                print()

                X = model.convert_board_to_input(board)
                y = np.zeros((1, 9))
                y[0, position] = 1

                model.train(X, y, epochs=1)
                break
            elif empty not in board:
                draw_board(board)
                print()
                print("게임 종료! 무승부입니다.")
                print()
                X = model.convert_board_to_input(board)
                y = np.zeros((1, 9))

                model.train(X, y, epochs=1)
                break
            else:
                current_player = 'O' if current_player == 'X' else 'X'

        iteration = iteration + 1

        if iteration >= train_itr:
            print(f"{iteration}회 학습이 완료되었습니다.")
        else:
            print(f"{iteration}회 학습 진행중...")

        if iteration >= train_itr:
            train_again = input("추가 학습을 하겠습니까? (y/n): ")
            if train_again.lower() == 'y':
                more_itr = input("추가로 몇 회 학습시키겠습니까?: ")
                if int(more_itr) > 0:
                    train_itr = train_itr + int(more_itr)
                    print()
                    print(f"{more_itr}회(총{iteration}회) 추가 학습 진행중...")
                    play_again = 'y'
                else:
                    print("잘못입력했습니다.")
                    print()
                    play_again = input("새 게임을 시작하시겠습니까? (y/n): ")
            else:
                print()
                play_again = input("새 게임을 시작하시겠습니까? (y/n): ")
        else:
            play_again = 'y'
        if play_again.lower() != 'y':
            break


######


play_game()


######