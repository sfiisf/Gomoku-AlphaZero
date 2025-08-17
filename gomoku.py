import numpy as np

class GomokuGame:
    def __init__(self, size=15):
        self.board_size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1

    def clear(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1

    def get_valid_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def is_move_out_of_bounds(self, x, y):
        return x < 0 or x >= self.board_size or y < 0 or y >= self.board_size

    def check_game_state(self):
        '''
        查询游戏状态
        返回值:
            0 游戏未结束
            1 先手玩家获胜
           -1 后手玩家获胜
            2 平局
        '''
        space = self.board_size * self.board_size
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != 0:
                    space -= 1
                    for x, y in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        cnt = 1
                        for k in range(1, 5):
                            ni = i + x * k
                            nj = j + y * k
                            if self.is_move_out_of_bounds(ni, nj) or self.board[i][j] != self.board[ni][nj]:
                                break
                            else:
                                cnt += 1
                            
                            if cnt == 5:
                                return self.board[i][j]
        if space > 0:
            return 2
        else:
            return 0
        
    def move(self, x, y):
        if self.is_move_out_of_bounds(x, y) or self.check_game_state() != 0 or self.board[x][y] != 0:
            return False
        self.board[x][y] = self.current_player
        self.current_player = -self.current_player
        return True

    def display(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print('  ' + ' '.join(str(i) for i in range(self.board_size)))
        for r in range(self.board_size):
            print(f'{r} ' + ' '.join(symbols[self.board[r][c]] for c in range(self.board_size)))
        print()