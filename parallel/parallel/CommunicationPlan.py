import numpy as np
from typing import List, Tuple, Callable, Optional
import logging
logger = logging.getLogger(__name__)
######
"""
Communication planning

Goal: come up with a near optimal communication schedule 

Input: Directed weighted communication graph, weights can be tuples, e.g
Output: schedule of communication (2D array, rows=rounds)

Algroithm: 
1) directed weighted graph -> undirected weighted graph
2) sort by message count (better balancing)
3) drop message counts
4) sort by degree of nodes in graph (doesnt this destroy the previous ordering?)
5) pick greedy from left to right


ToDo check this later, for now just take workin version
One should maybe also check the balancing and give some diagnostics
"""
######


# helper funtion for testing
def neighbor_check(i: int, j: int, n: int = 4):
    if i == j:
        return False
    else:
        m = min(i, j)
        M = max(i, j)
        d = M - m
        if d == 1:  # 1 left/right
            return True
        elif d == n:  # n up/down
            return True
        elif d == n - 1 or d == n + 1:
            if not m // n == M // n:
                return True  # n-1, n+1 diagonals
        else:
            pass
    return False
# just for testing
def get_random_comms(
    neighbor_condition: Callable[[int, int], bool] = neighbor_check,
    seed: int | float = 0,
    n=4,
) -> np.ndarray:
    np.random.seed(seed)
    comms = []
    N_procs = n * n
    for i in range(N_procs):
        for j in range(N_procs):
            if neighbor_condition(i, j, n):
                rands = np.random.randint(0, [1000, 100, 50])
                comms.append([i, j, *rands])
    return np.array(comms, dtype=np.int32)


def combine_directed_comms(directed_comms: np.ndarray) -> np.ndarray:
    undirected_graph = {}
    for message in directed_comms:
        comm_partners = [message[0], message[1]]
        comm_adress = (min(comm_partners), max(comm_partners))
        if comm_adress in undirected_graph:
            undirected_graph[comm_adress] += message[2:]
        else:
            undirected_graph[comm_adress] = message[2:]
    undirected_comms = []
    for key, value in undirected_graph.items():
        undirected_comms.append([*key, *value])
    return np.array(undirected_comms, dtype=np.int32)


def sort_by_message_count(undirected_comms: np.ndarray) -> np.ndarray:
    return np.array(
        sorted(undirected_comms, key=lambda x: sum(x[2:]), reverse=True), dtype=np.int32
    )


def drop_message_count(comms: np.ndarray):
    return np.array([message[:2] for message in comms], dtype=np.int32)


def max_adress(adresses: np.ndarray) -> int:
    return np.max([max(a[0], a[1]) for a in adresses])


def greedy_schedule(
    sorted_adresses: List[Tuple[int]], N: Optional[int] = None
) -> List[List[Tuple[int]]]:
    if N is None:
        N = max_adress(sorted_adresses) + 1
    working_copy = sorted_adresses.copy()
    comm_rounds = []
    while len(working_copy) > 0:
        counter = np.zeros(N, dtype=np.int32)
        new_round = []
        indices_new_round = []
        for i, adresses in enumerate(working_copy):
            if max(counter[[adresses[0], adresses[1]]]) == 0:
                indices_new_round.append(i)
                new_round.append(adresses)
                counter[[adresses[0], adresses[1]]] += 1

        comm_rounds.append(new_round)
        for index in sorted(indices_new_round, reverse=True):
            del working_copy[index]
    return comm_rounds


def min_rounds_schedule(
    adresses: List[Tuple[int]], N: Optional[int] = None
) -> List[List[Tuple[int]]]:
    if N is None:
        N = max_adress(adresses) + 1
    counter = np.zeros(N, dtype=np.int32)
    for s, r in adresses:
        counter[[s, r]] += 1
    sorted_adresses = sorted(
        adresses, key=lambda x: max(counter[x[0]], counter[x[1]]), reverse=True
    )
    return greedy_schedule(sorted_adresses, N=N)


def schedule_from_comms(comms: np.ndarray) -> List[List[np.ndarray]]:
    undirected_comms = combine_directed_comms(comms)
    sorted_comms = sort_by_message_count(undirected_comms)
    dropped_comms = drop_message_count(sorted_comms)
    schedule = min_rounds_schedule(dropped_comms)
    return schedule


def flatten_schedule(schedule: List[List[np.ndarray]]) -> np.ndarray:
    flattened_schedule = [len(schedule)]
    flattened_tail = []
    for round in schedule:
        flattened_tail.append(np.array(round).flatten())
        flattened_schedule.append(len(round))
    flattened_schedule = np.concatenate(
        (np.array(flattened_schedule, dtype=np.int32), *flattened_tail)
    )
    return flattened_schedule


def unflatten_schedule(flattened_schedule: np.ndarray) -> List[List[np.ndarray]]:
    num_rounds = flattened_schedule[0]
    len_rounds = flattened_schedule[1 : num_rounds + 1]
    current_index = num_rounds + 1
    schedule = []
    for i in range(num_rounds):
        round = []
        for j in range(len_rounds[i]):
            round.append(flattened_schedule[current_index : current_index + 2])
            current_index += 2
        schedule.append(round)
    return schedule


if __name__ == "__main__":
    comms = get_random_comms()
    undirected_comms = combine_directed_comms(comms)
    print(undirected_comms)
    schedule = schedule_from_comms(comms)
    print("== communication schedule")
    flattened = flatten_schedule(schedule)
    print(flattened)
    unflattened = unflatten_schedule(flattened)
    print(flatten_schedule(unflattened))
    for i, round in enumerate(unflattened):
        print(i, "#", *np.array(round))
