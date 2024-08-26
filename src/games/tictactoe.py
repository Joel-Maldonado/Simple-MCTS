import numpy as np
from typing import Tuple


class TicTacToe:
    """
    A class representing the game of Tic-Tac-Toe.
    All methods are static, providing functionality for game state management.
    """

    @staticmethod
    def get_initial_state() -> np.ndarray:
        """
        Initialize an empty Tic-Tac-Toe board.

        Returns:
            np.ndarray: An empty game board as a 2D numpy array.
        """
        return np.zeros((3, 3), dtype=int)

    @staticmethod
    def get_next_state(state: np.ndarray, action: int, player: int) -> np.ndarray:
        """
        Apply an action to the current game state.

        Args:
            state (np.ndarray): The current game state.
            action (int): The action to take, represented as a flattened index.
            player (int): The player taking the action (1 or -1).

        Returns:
            np.ndarray: The new game state after applying the action.
        """
        size = state.shape[0]
        row, col = action // size, action % size
        new_state = state.copy()
        new_state[row, col] = player
        return new_state

    @staticmethod
    def get_valid_actions(state: np.ndarray) -> np.ndarray:
        """
        Get all valid actions for the current game state.

        Args:
            state (np.ndarray): The current game state.

        Returns:
            np.ndarray: An array of valid actions (flattened indices).
        """
        return np.where(state.reshape(-1) == 0)[0]

    @staticmethod
    def check_win(state: np.ndarray, last_action: int) -> bool:
        """
        Check if the last action resulted in a win.

        Args:
            state (np.ndarray): The current game state.
            action (int): The last action taken.

        Returns:
            bool: True if the last action resulted in a win, False otherwise.
        """
        size = state.shape[0]
        row, col = last_action // size, last_action % size
        player = state[row, col]

        # Check row, column, and diagonals
        return (
            np.sum(state[row, :]) == player * size
            or np.sum(state[:, col]) == player * size
            or np.sum(np.diag(state)) == player * size
            or np.sum(np.diag(np.fliplr(state))) == player * size
        )

    @staticmethod
    def is_terminal(state: np.ndarray, last_action: int = None) -> bool:
        """
        Check if the game has ended.

        Args:
            state (np.ndarray): The current game state.
            action (int): The last action taken.

        Returns:
            bool: True if the game has ended, False otherwise.
        """
        if last_action is None:
            return False
        return (
            TicTacToe.check_win(state, last_action)
            or len(TicTacToe.get_valid_actions(state)) == 0
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
        if TicTacToe.check_win(state, action):
            return 1
        if len(TicTacToe.get_valid_actions(state)) == 0:
            return 0.5
        return -1

    @staticmethod
    def change_perspective(state: np.ndarray, player: int) -> np.ndarray:
        """
        Change the perspective of the board to the given player.

        Args:
            state (np.ndarray): The current game state.
            player (int): The player whose perspective to change to (1 or -1).

        Returns:
            np.ndarray: The game state from the perspective of the given player.
        """
        return state * player

    @staticmethod
    def visualize_state(state: np.ndarray) -> None:
        """
        Visualize the current state of the game board.

        Args:
            state (np.ndarray): The current game state.
        """
        size = state.shape[0]
        symbols = {1: "X", -1: "O", 0: " "}

        for i in range(size):
            row = " | ".join(symbols[cell] for cell in state[i])
            print(row)
            if i < size - 1:
                print("-" * (4 * size - 3))
