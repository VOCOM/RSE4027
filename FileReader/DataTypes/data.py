from tabulate import tabulate

class DATA:
    def __init__(self) -> None:
        # TODO: Change DATA to dictionary for [key : value] pair -> [str : array]
        self.header = []
        self.data = []
        
    # TODO
    def AppendData(self):
        pass

    def PrintData(self):
        print(tabulate(self.data, headers=self.header))