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

    def step(self, action: int) -> tuple:
        if action not in self.action_space:
            raise ValueError(f"No action {action}")
        
        step_row, step_col = self.action_map[action]
        row, col = self.current_state

        next_row = self.current_state[0] + step_row
        next_col = self.current_state[1] + step_col

        if 0 <= row + step_row < self.height or 0 <= col + step_col < self.width:
            next_state = (next_row, next_col)
        else:
            next_state = self.current_state    
            
        self.current_state = next_state
        
        # the new state we will be in
        reward, done = self.calc_reward(next_state)

        return next_state, reward, done

    def calc_reward(self, next_state):
        if next_state == self.end_state:
            reward = 10
            done = True
        elif next_state in self.obstacles:
            reward = -5
            done = True
        else:
            reward = -1
            done = False
        return reward,done


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
    grid.step(3)
    grid.step(3)
    grid.render()
    grid.step(3)
    grid.render()
    grid.step(3)
    grid.render()