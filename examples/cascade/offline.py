import libem
from libem.prepare.datasets import abt_buy
from benchmark.run import args
from libem.cascade.function import offline


def main():
    offline(args(), abt_buy)


if __name__ == '__main__':
    main()