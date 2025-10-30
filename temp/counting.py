import jax
import random

class Counter():
    def __init__(self):
        self.x = 0


def increment_counter(counter, by):
    counter.x+=by

@jax.jit
def f():
    counter = Counter() 
    increment_counter(counter, 10)
    increment_counter(counter, 12123)
    increment_counter(counter, 3)
    print(f"DEBUGPRINT[36]: counting.py:14: counter={counter}")
    return counter.x


a =  f()
print(f"DEBUGPRINT[39]: counting.py:23: a={a}")
a =  f()
print(f"DEBUGPRINT[40]: counting.py:25: a={a}")


