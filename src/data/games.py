import random
import itertools
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset, DataLoader


MATH_GAME_OPERATIONS = {
    "sum": lambda x, y: x + y,
    "mod": lambda x, y: x % y,
    "xor": lambda x, y: x ^ y,
    "and": lambda x, y: x & y,
    "rsh": lambda x, y: x >> y
}


@dataclass
class MathGameConfig():
    name: str = "sum-mod-3"
    seed: int = 0
    P: int = 113
    max: int = 113
    min: int = 0
    test_ratio: float = 0.7
    P_gap: int = 1


class MathGameDataset():
    """
    Dataset of toy math games for grokking.
    inputs: 3 tokens [A, B, P]
    targets: op2( op1(A, B) , P )
    """
    def __init__(self, cfg: MathGameConfig):
        random.seed(cfg.seed)
        checked_cfg = deepcopy(cfg)
        [op1, op2, N] = checked_cfg.name.split("-")

        self.num_input_tokens = N
        self.op1_id = op1
        self.op2_id = op2

        self.op1 = MATH_GAME_OPERATIONS[op1]
        self.op2 = MATH_GAME_OPERATIONS[op2]

        self.cfg = checked_cfg
        self.name = checked_cfg.name

    def create_problem_set(self, split_train_test: bool =True):
        problems = []
        answers = []
        
        offset = 1 - self.cfg.P_gap
        for A in range(self.cfg.min, self.cfg.max+offset):
            for B in range(self.cfg.min, self.cfg.max+offset):
                problems.append([A, B, self.cfg.P])
                answers.append(self.op2(self.op1(A, B), self.cfg.P))

        problem_set_size = len(problems)
        indices = [i for i in range(problem_set_size)]

        # shuffle for random ordering
        random.shuffle(indices)

        problems_np = torch.tensor([problems[idx] for idx in indices], dtype=torch.long)
        answers_np = torch.tensor([answers[idx] for idx in indices], dtype=torch.long)

        if not split_train_test:
            return problems_np, answers_np

        num_test_problems = int(self.cfg.test_ratio * problem_set_size)
        num_train_problems = problem_set_size - num_test_problems

        train_problems = (
            problems_np[:num_train_problems, :],
            answers_np[:num_train_problems]
        )
        test_problems = (
            problems_np[-num_test_problems:, :],
            answers_np[-num_test_problems:]
        )
        # print(f"Created {self.cfg.name.upper()} game with P={self.cfg.P}")
        # print(f"no. of total problems={problem_set_size}, no. of train problems={num_train_problems}, no. of test problems={num_test_problems}")
        return train_problems, test_problems

    def create_minibatched_problem_set(self, batch_size: int, split_train_test: bool = True, num_workers: int = 1):
        train_problems, test_problems = self.create_problem_set(split_train_test)
        train_dataset, test_dataset = TensorDataset(*train_problems), TensorDataset(*test_problems)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False)
        
        return itertools.cycle(train_loader), itertools.cycle(test_loader)


def test_math_games():
    cfg = MathGameConfig(
        name="sum-mod-3",
        seed=0,
        P=113,
        max=113,
        min=0,
        test_ratio=0.7
    )
    dataset = MathGameDataset(cfg)
    train_dataset, test_dataset = dataset.create_problem_set()
    X_train, y_train = train_dataset
    print(X_train.shape, y_train.shape)
    print(X_train[0], y_train[0])
    print(" ")

    X_test, y_test = test_dataset
    print(X_test.shape, y_test.shape)
    print(X_test[0], y_test[0])

    train_loader, test_loader = dataset.create_minibatched_problem_set(batch_size=32)
    train_loader = iter(train_loader)
    
    for i in range(3):
        train_tokens, train_labels = next(train_loader)
        print(train_tokens.shape, train_labels.shape)
        print(train_tokens[i])
        print(" ")


if __name__ == "__main__":
    test_math_games()
