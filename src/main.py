import numpy as np
from typing import Optional, List
from numba import njit
import importlib

class Node:
    """
    A class representing a node in the Monte Carlo Tree Search.
    """

    def __init__(
        self,
        state: np.ndarray,
        game,
        parent: Optional["Node"] = None,
        action_taken: Optional[int] = None,
    ) -> None:
        """
        Initialize a new Node.

        Args:
            state (np.ndarray): The game state at this node.
            game: The game module (either TicTacToe or Connect4).
            parent (Optional[Node]): The parent node.
            action_taken (Optional[int]): The action that led to this node.
        """
        self.state = state
        self.game = game
        self.parent = parent
        self.action_taken = action_taken
        self.children: List[Node] = []
        self.num_visits = 0
        self.num_wins = 0
        self.C = np.sqrt(2)
        self.actions_available = game.get_valid_actions(self.state)

    def uct(self) -> float:
        """
        Calculate the Upper Confidence Bound for Trees (UCT) value for this node.

        Returns:
            float: The UCT value.
        """
        return uct(self.num_wins, self.num_visits, self.parent.num_visits, self.C)

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible child nodes have been expanded.

        Returns:
            bool: True if fully expanded, False otherwise.
        """
        return len(self.actions_available) == 0 and len(self.children) > 0

    def best_child(self) -> "Node":
        """
        Select the child node with the highest UCT value.

        Returns:
            Node: The best child node.
        """
        return max(self.children, key=lambda child: child.uct())

    def expand(self) -> "Node":
        """
        Expand this node by adding a new child node.

        Returns:
            Node: The newly created child node.
        """
        action = np.random.choice(self.actions_available)
        self.actions_available = np.delete(
            self.actions_available, np.where(self.actions_available == action)
        )

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, -1)

        child = Node(child_state, self.game, parent=self, action_taken=action)
        self.children.append(child)
        return child

    def simulate(self) -> float:
        """
        Simulate a random playout from this node until a terminal state is reached.

        Returns:
            float: The outcome of the simulation.
        """
        if self.game.is_terminal(self.state, self.action_taken):
            return self.game.get_outcome(self.state, self.action_taken)

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            action = np.random.choice(self.game.get_valid_actions(rollout_state))
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)

            if self.game.is_terminal(rollout_state, action):
                return self.game.get_outcome(rollout_state, action)

            rollout_player = -rollout_player

@njit(cache=True)
def uct(num_wins: int, num_visits: int, parent_visits: int, C: float):
    if num_visits == 0:
        return np.inf
    return num_wins / num_visits + C * np.sqrt(np.log(parent_visits) / num_visits)

def MCTS(state: np.ndarray, num_simulations: int, game) -> int:
    """
    Perform Monte Carlo Tree Search to find the best action.

    Args:
        state (np.ndarray): The current game state.
        num_simulations (int): The number of simulations to run.
        game: The game module (either TicTacToe or Connect4).

    Returns:
        int: The best action to take.
    """
    root = Node(state, game)

    for _ in range(num_simulations):
        node = root

        # Selection
        while node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not game.is_terminal(node.state, node.action_taken):
            node = node.expand()

        # Simulation
        outcome = node.simulate()

        # Backpropagation
        while node:
            node.num_visits += 1
            node.num_wins += outcome
            outcome = 1 - outcome
            node = node.parent

    return max(root.children, key=lambda c: c.num_visits).action_taken

def play_game(game, num_simulations: int) -> None:
    """
    Play a game with an MCTS bot.

    Args:
        game: The game module (either TicTacToe or Connect4).
    """
    state = game.get_initial_state()
    player = 1

    print(f"You are player 1, and the MCTS bot is player -1.\n")
    
    while True:
        game.visualize_state(state)

        if player == 1:
            valid_moves = game.get_valid_actions(state)
            print(f"Valid moves: {valid_moves}")
            action = input(f"{player}: ")
            if action.isdigit():
                action = int(action)
            else:
                print("Invalid input!")
                continue


            if action not in valid_moves:
                print("Action not valid!")
                continue
        else:
            neutral_state = game.change_perspective(state, player)
            action = MCTS(neutral_state, num_simulations, game)
            print(f"\nMCTS: {action}")

        state = game.get_next_state(state, action, player)
        outcome = game.get_outcome(state, action)

        # Game Over
        if outcome != -1:
            game.visualize_state(state)
            if outcome == 1:
                if player == 1:
                    print("Player 1 has won!")
                else:
                    print("MCTS has won!")
            else:
                print("Draw!")
            break

        player = -player

def main():
    while True:
        game_choice = input("Choose a game to play (1 for Tic-Tac-Toe, 2 for Connect4): ").strip()
        if game_choice == '1':
            game = importlib.import_module('games.tictactoe').TicTacToe
            break
        elif game_choice == '2':
            game = importlib.import_module('games.connect4').Connect4
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    
    num_simulations = input("Enter the number of simulations for MCTS (Default 1000): ")
    if num_simulations.isdigit():
        num_simulations = int(num_simulations)
    else:
        num_simulations = 1000
        print("Using default value of 1000 simulations")

    play_game(game, num_simulations)

if __name__ == "__main__":
    main()