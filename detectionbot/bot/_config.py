import argparse


class Parser:
    """
    Parser class for accesing parameters when called `python run.py <TOKEN>`
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('TOKEN', nargs=1)

    @property
    def parse_args(self) -> str:
        """
        Property that gives str-like object of TOKEN
        :return: str
        """
        return self.parser.parse_args().TOKEN[0]
