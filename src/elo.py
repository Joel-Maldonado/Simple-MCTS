from src.main import MCTS
from src.games.tictactoe import TicTacToe as t
from collections import defaultdict
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class EloRanking:
    def __init__(self, initial_elo=1200, k_factor=32):
        self.elo_ratings = defaultdict(lambda: initial_elo)
        self.k_factor = k_factor

    def update_elo(self, player_a, player_b, outcome_a):
        rating_a = self.elo_ratings[player_a]
        rating_b = self.elo_ratings[player_b]

        expected_a = 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        self.elo_ratings[player_a] += self.k_factor * (outcome_a - expected_a)
        self.elo_ratings[player_b] += self.k_factor * ((1 - outcome_a) - expected_b)

    def get_elo(self, player):
        return self.elo_ratings[player]


def run_match(mcts_1, mcts_2, simulations_1, simulations_2, board_size=3):
    state = t.get_initial_state(board_size)
    player = 1

    while True:
        if player == 1:
            action = mcts_1(state, simulations_1)
        else:
            neutral_state = t.change_perspective(state, player)
            action = mcts_2(neutral_state, simulations_2)

        state = t.get_next_state(state, action, player)
        outcome = t.get_outcome(state, action)

        if outcome == 1:
            return player
        elif outcome == 0.5:
            return 0.5

        player = -player


def process_matches(args):
    bot_1_name, bot_2_name, mcts_1, mcts_2, simulations_1, simulations_2 = args
    result = run_match(mcts_1, mcts_2, simulations_1, simulations_2)
    return bot_1_name, bot_2_name, result


def mcts_1_simulation(state, _):
    return MCTS(state, 1)


def mcts_50_simulations(state, _):
    return MCTS(state, 50)


def mcts_100_simulations(state, _):
    return MCTS(state, 100)


def mcts_200_simulations(state, _):
    return MCTS(state, 200)


def mcts_500_simulations(state, _):
    return MCTS(state, 500)


def mcts_1000_simulations(state, _):
    return MCTS(state, 1000)


if __name__ == "__main__":
    elo_ranking = EloRanking()
    mcts_versions = {
        "MCTS_1_simulation": mcts_1_simulation,
        "MCTS_50_simulations": mcts_50_simulations,
        "MCTS_100_simulations": mcts_100_simulations,
        "MCTS_200_simulations": mcts_200_simulations,
    }

    games = 1000
    total_matches = len(mcts_versions) * (len(mcts_versions) - 1) * games
    tasks = []

    for bot_1_name, bot_1 in mcts_versions.items():
        for bot_2_name, bot_2 in mcts_versions.items():
            if bot_1_name != bot_2_name:
                for _ in range(games):  # Prepare 100 games
                    tasks.append(
                        (bot_1_name, bot_2_name, bot_1, bot_2, 50, 100)
                    )  # Modify as needed

    with ProcessPoolExecutor() as executor:
        with tqdm(
            total=total_matches, desc="Running Elo Tournament", unit="match"
        ) as pbar:
            futures = [executor.submit(process_matches, task) for task in tasks]
            for future in as_completed(futures):
                bot_1_name, bot_2_name, result = future.result()
                if result == 1:
                    elo_ranking.update_elo(bot_1_name, bot_2_name, 1)
                elif result == -1:
                    elo_ranking.update_elo(bot_1_name, bot_2_name, 0)
                else:
                    elo_ranking.update_elo(bot_1_name, bot_2_name, 0.5)
                pbar.update(1)

    # Print the final Elo ratings
    for bot_name in mcts_versions.keys():
        print(f"{bot_name}: {elo_ranking.get_elo(bot_name)}")
