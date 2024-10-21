import os
import random


def gen_log(
    L: int = 10000,
    S: int = 10,
    V: int = 10,
    C: int = 10,
    filename: str = "test.log",
    seed: int = 42,
):
    """
    Generate a log for the microbenchmarks, as per the LOGos paper.

    Parameters:
        L: The number of lines in the log.
        S: The number of distinct values for the string token.
        V: The number of numerical variables.
        C: The number of numerical constants.
        filename: The filename to write the log to.
    """

    # Generate path to filename if it doesn't exist
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    prng = random.Random(seed)

    with open(filename, "w+") as f:
        for i in range(L):
            f.write(f"line_{i}")
            f.write(f" s_{prng.randint(1,S)}")
            for j in range(V):
                f.write(f" {i}")
            for j in range(C):
                f.write(f" 0")
            f.write("\n")
