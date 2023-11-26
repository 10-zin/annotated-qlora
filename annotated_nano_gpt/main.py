from nano_gpt import GPT
import time


def main():
    start = time.perf_counter()
    model = GPT()
    print(time.perf_counter() - start)
    print(model)


if __name__ == "__main__":
    main()
