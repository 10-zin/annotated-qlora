import os
import random

import numpy as np
from nano_gpt import GPT
import time
import torch
from config import Config


def seed_everything(seed=0):
    """Seed everything."""
    # PyTorch random seed
    torch.manual_seed(seed)

    # If using CUDA (PyTorch with GPU support)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
        # The following two settings are recommended for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Python's random module
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # Set the environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    
    config = Config()
    seed_everything(42)
    # start = time.perf_counter()
    model = GPT(config)
    # print(f"\nTime to build GPT Architecture: {(time.perf_counter() - start):.4f}s \n\n")
    # print(model)
    # Call the function with your desired seed
    
    rand_inp = torch.randint(1, 100, size=[2, 21])
    print(rand_inp)
    # exit()
    # torch.manual_seed(0)
    # rand_inp2 = torch.randint(1, 100, size=[2, 21])
    # assert (rand_inp==rand_inp2).all()
    # exit()
    # assert (rand_inp == rand_inp2).all()

    # print(rand_inp)
    start = time.perf_counter()
    op = model(rand_inp)
    print(f"\nTime to pass inp: {(time.perf_counter() - start):.4f}s \n\n")
    print(op)
    print(op.shape)


if __name__ == "__main__":
    main()
