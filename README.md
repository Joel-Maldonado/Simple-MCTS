# Monte Carlo Tree Search (MCTS) for Tic-Tac-Toe

This project implements a basic Monte Carlo Tree Search (MCTS) algorithm for playing Tic-Tac-Toe. It's primarily designed as an educational tool to demonstrate the fundamental concepts of a simple MCTS algorithm.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Monte Carlo Tree Search Algorithm](#monte-carlo-tree-search-algorithm)
3. [Results](#simple-results)
4. [Going Further](#going-further)

## Project Overview

This project is an experimental implementation of the Monte Carlo Tree Search algorithm applied to the game of Tic-Tac-Toe. It's important to note that this is a learning exercise and not an optimized or highly competitive AI. The main goals of this project are:

1. To demonstrate how random playouts can lead to intelligent decision-making through the MCTS algorithm.
2. To serve as a starting point for further experimentation and improvement.

The implementation uses the standard, vanilla Monte Carlo Tree Search algorithm without any advanced optimizations or enhancements. This choice was made to clearly illustrate the core principles of MCTS.

## Monte Carlo Tree Search Algorithm

MCTS is a heuristic search algorithm for some kinds of decision processes, most notably those employed in game play. The algorithm combines the generality of random simulation with the precision of tree search.

The MCTS algorithm consists of four main steps:

1. **Selection**: Starting from the root node, select successive child nodes down to a leaf node. The selection of child nodes is based on the Upper Confidence Bound for Trees (UCT) formula:

   $$UCT = \frac{w_i}{n_i} + C\sqrt{\frac{\ln(N_i)}{n_i}}$$

   Where:
   - $w_i$ = Number of wins after the i-th move
   - $n_i$ = Number of simulations after the i-th move
   - $N_i$ = Total number of simulations for the parent node
   - $C$ = Exploration parameter

   Note: The exploration parameter $C$ is usually set to $\sqrt{2}$, as this value exhibits asymptotic optimality when rewards are between 0 and 1. In our implementation, we use this standard value.

2. **Expansion**: If the selected node is not terminal and not fully expanded, create one or more child nodes.

3. **Simulation**: Perform a random playout from the new node(s) until a terminal state is reached.

4. **Backpropagation**: Update the node statistics (visit count and win count) for all nodes in the path from the new node to the root.

### Advantages of Monte Carlo Tree Search

1. **Domain-independent**: MCTS can be applied to various games and problem domains without extensive domain-specific knowledge.
2. **Anytime algorithm**: It can return a valid result even if interrupted before completion, with the quality of the result improving given more computation time.
3. **Asymmetric tree growth**: MCTS focuses computational resources on the most promising lines of play branching out significantly less than algorithms like mini-max.
4. **No need for a heuristic evaluation function**: Unlike traditional minimax algorithms, MCTS doesn't require a complex evaluation function for non-terminal states.

### Limitations of This Implementation

1. **Simple rollout policy**: Our implementation uses random playouts, which may not be optimal for complex games.
2. **Simplicity of Tic-Tac-Toe**: Tic-Tac-Toe has a very small action space, and thus doesn't make very much use of MCTS's asymmetric tree growth as much as other games with a large action space.
3. **Computationally intensive**: Achieving strong play often requires a large number of simulations, which can be time-consuming.

## Project Structure

The project consists of three main Python files:

1. `main.py`: Contains the MCTS implementation and the main game loop.
2. `tictactoe.py`: Defines the Tic-Tac-Toe game rules and board representation.
3. `elo.py`: Implements the Elo ranking system and runs tournaments between different MCTS versions.

## Simple Results

After running 10,000 games in the Elo tournament, the following ratings were obtained using no evaluation function or heuristic other than random moves:

```
Baseline (Random Moves without MCTS): 583.0899959499717
MCTS_50_simulations: 1102.4090984116353
MCTS_100_simulations: 1296.5373273412813
MCTS_200_simulations: 1242.2460519124993
MCTS_500_simulations: 1422.9611929410578
MCTS_1000_simulations: 1552.7563334435456
```

These results show that increasing the number of simulations generally improves the performance of the MCTS algorithm, with diminishing returns as the simulation count gets very high.

---

## Going Further

While this project serves as a good introduction to MCTS, there are several ways it could be improved:

1. **Evaluation Function**: Implementing a heuristic evaluation function for non-terminal states could significantly improve the quality of playouts and overall performance.

2. **Neural Network Integration**: Following the approach of algorithms like AlphaZero, incorporating a neural network for both policy and value estimation could drastically improve the AI's playing strength and efficiency.

3. **Parallelization**: The current implementation could be optimized to take better advantage of parallel processing, allowing for more simulations in less time.

4. **Progressive Widening**: For games with larger branching factors, techniques like progressive widening could be implemented to manage the expansion of nodes more efficiently.

5. **Domain-Specific Heuristics**: For Tic-Tac-Toe specifically, incorporating simple heuristics (like prioritizing center and corner moves) could improve performance without compromising the educational value of the base MCTS implementation.

6. **RAVE (Rapid Action Value Estimation)**: This enhancement to MCTS could provide faster convergence to good policies in many games.

These potential improvements showcase the flexibility of MCTS and provide avenues for further learning and experimentation. You are encouraged to try implementing these enhancements as exercises to deepen understanding
