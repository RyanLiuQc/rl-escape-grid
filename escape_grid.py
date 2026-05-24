import numpy as np 

class Grid:
    def __init__(self):
        self.grid = np.zeros(shape=(5,5))
        print("5x5 grid created", self.grid)
        

if __name__ == "__main__":
    grid = Grid()