from typing import Protocol
import trackio
import jax


class Logger(Protocol):
    def log(self, *args, **kwargs):
        ...

class DummyLogger(Logger):
    def __init__(*args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        return 

