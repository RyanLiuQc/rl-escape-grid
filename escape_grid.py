import numpy as np 

class Grid:
    def __init__(self, obstacle_pos=None):
        self.height = 5
        self.width = 5

        #print("5x5 grid created", self.grid)

        # add obstables
        if not obstacle_pos:
            self.obstacles = {(2,2), (0,2), (3,2), (3,1), (4,3)}
        else:
            self.obstacles = set(obstacle_pos)
        


        self.start_state = (4,0)
        self.end_state = (0,4)

        self.current_state = self.start_state

        self.action_space = [0,1,2,3]
        self.action_map = {
            0: (-1, 0), # up
            1: (1, 0), # down
            2: (0, -1), # left 
            3: (0, 1) # right
        }

    def reset(self) -> None:
        pass

    def step(self):
        pass

    # improve render 
    def render(self):
        grid = np.full((self.height, self.width), fill_value=". ", dtype=object)
        grid[self.current_state] = "X "
        grid[self.end_state] = "G "

        for obs in self.obstacles:
            grid[obs] = "T "
        
        print("\n" + "-" * 15)
        for row in grid:
            print("".join(row))
        print("-" * 15)

        

if __name__ == "__main__":
    grid = Grid()
    grid.render()