import numpy as np
from typing import Optional, List
from tictactoe import TicTacToe as t


class Node:
    """
    A class representing a node in the Monte Carlo Tree Search.
    """

    def __init__(
        self,
        state: np.ndarray,
        parent: Optional["Node"] = None,
        action_taken: Optional[int] = None,
    ) -> None:
        """
        Initialize a new Node.

        Args:
            state (np.ndarray): The game state at this node.
            parent (Optional[Node]): The parent node.
            action_taken (Optional[int]): The action that led to this node.
        """
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: List[Node] = []
        self.num_visits = 0
        self.num_wins = 0
        self.C = np.sqrt(2)
        self.actions_available = t.get_valid_actions(self.state)

    def uct(self) -> float:
        """
        Calculate the Upper Confidence Bound for Trees (UCT) value for this node.

        Returns:
            float: The UCT value.
        """
        if self.num_visits == 0:
            return float("inf")
        return self.num_wins / self.num_visits + self.C * np.sqrt(
            np.log(self.parent.num_visits) / self.num_visits
        )

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

        child_state = t.get_next_state(self.state, action, 1)
        child_state = t.change_perspective(child_state, -1)

        child = Node(child_state, parent=self, action_taken=action)
        self.children.append(child)
        return child

    def simulate(self) -> float:
        """
        Simulate a random playout from this node until a terminal state is reached.

        Returns:
            float: The outcome of the simulation.
        """
        if t.is_terminal(self.state, self.action_taken):
            return t.get_outcome(self.state, self.action_taken)

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            action = np.random.choice(t.get_valid_actions(rollout_state))
            rollout_state = t.get_next_state(rollout_state, action, rollout_player)

            if t.is_terminal(rollout_state, action):
                return t.get_outcome(rollout_state, action)

            rollout_player = -rollout_player


def MCTS(state: np.ndarray, num_simulations: int) -> int:
    """
    Perform Monte Carlo Tree Search to find the best action.

    Args:
        state (np.ndarray): The current game state.
        num_simulations (int): The number of simulations to run.

    Returns:
        int: The best action to take.
    """
    root = Node(state)

    for _ in range(num_simulations):
        node = root

        # Selection
        while node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not t.is_terminal(node.state, node.action_taken):
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


def play_game() -> None:
    """
    Play a game of Tic-Tac-Toe with an MCTS bot.
    """
    state = t.get_initial_state(3)
    player = 1

    print("You are player 1, and the MCTS bot is player -1.\n")
    
    while True:
        print(state)

        if player == 1:
            valid_moves = t.get_valid_actions(state)
            print(f"Valid moves: {valid_moves}")
            action = int(input(f"{player}: "))

            if action not in valid_moves:
                print("Action not valid!")
                continue
        else:
            neutral_state = t.change_perspective(state, player)
            action = MCTS(neutral_state, 1000)
            print(f"\nMCTS: {action}")

        state = t.get_next_state(state, action, player)
        outcome = t.get_outcome(state, action)

        # Game Over
        if outcome != -1:
            print(state)
            if outcome == 1:
                if player == 1:
                    print("Player 1 has won!")
                else:
                    print("MCTS has won!")
            else:
                print("Draw!")
            break

        player = -player


if __name__ == "__main__":
    play_game()
