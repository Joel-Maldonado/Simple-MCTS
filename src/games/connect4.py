import numpy as np
from numba import njit

class Connect4:
    @staticmethod
    def get_initial_state() -> np.ndarray:
        return np.zeros((6, 7), dtype=int)

    @staticmethod
    @njit(cache=True)
    def _add_move(state: np.ndarray, action: int, player: int):
        for row in range(5, -1, -1):
            if state[row][action] == 0:
                state[row][action] = player
                break
        
    @staticmethod
    def get_next_state(state: np.ndarray, action: int, player: int) -> np.ndarray:
        valid_actions = Connect4.get_valid_actions(state)
        if action in valid_actions:
            Connect4._add_move(state, action, player)
        else:
            raise ValueError("Invalid action")
        return state

    @staticmethod
    @njit(cache=True)
    def get_valid_actions(state: np.ndarray) -> np.ndarray:
        return np.where(state[0] == 0)[0]

    @staticmethod
    @njit(cache=True)
    def check_win(state: np.ndarray, action: int) -> bool:
        rows, cols = state.shape
        row = np.argmax(state[:, action] != 0)
        player = state[row, action]

        # Define directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, action + dc * i
                if 0 <= r < rows and 0 <= c < cols and state[r, c] == player:
                    count += 1
                else:
                    break

            for i in range(1, 4):
                r, c = row - dr * i, action - dc * i
                if 0 <= r < rows and 0 <= c < cols and state[r, c] == player:
                    count += 1
                else:
                    break

            if count >= 4:
                return True

        return False

    @staticmethod
    def is_terminal(state: np.ndarray, last_action: int = None) -> bool:
        if last_action is None:
            return False
        return (
            Connect4.check_win(state, last_action)
            or len(Connect4.get_valid_actions(state)) == 0
        )
    
    @staticmethod
    def get_outcome(state: np.ndarray, action: int) -> float:
        """
        Get the outcome of the game.

        Args:
            state (np.ndarray): The current game state.
            action (int): The last action taken.

        Returns:
            float: 1 if the last player won, 0.5 for a draw, -1 if the game is not over.
        """
        if Connect4.check_win(state, action):
            return 1
        if len(Connect4.get_valid_actions(state)) == 0:
            return 0.5
        return -1

    @staticmethod
    def change_perspective(state: np.ndarray, player: int) -> np.ndarray:
        return state * player
    
    @staticmethod
    def visualize_state(state: np.ndarray) -> None:
        """
        Visualize the Connect4 board state in the console.

        Args:
            state (np.ndarray): The current game state.
        """
        print()
        for row in state:
            print("|", end="")
            for cell in row:
                if cell == 1:
                    print("\033[91m●\033[0m|", end="")  # Red circle
                elif cell == -1:
                    print("\033[93m●\033[0m|", end="")  # Yellow circle
                else:
                    print(" |", end="")
            print()
        print("-" * 15)
        print("|0|1|2|3|4|5|6|")
        print()


if __name__ == "__main__":
    state = Connect4.get_initial_state()
    player = 1

    while True:
        Connect4.visualize_state(state)

        valid_moves = Connect4.get_valid_actions(state)
        print(f"Valid moves: {valid_moves}")
        action = int(input(f"{player}: "))

        if action not in valid_moves:
            print("Action not valid!")
            continue

        state = Connect4.get_next_state(state, action, player)

        if Connect4.is_terminal(state, action):
            Connect4.visualize_state(state)
            outcome = Connect4.get_outcome(state, action)
            if outcome == 1:
                print("Player 1 wins!")
            elif outcome == 0.5:
                print("It's a draw!")
            else:
                print("Player -1 wins!")
            break
        
        player *= -1