from mesa import Agent, Model
import random
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128
import numpy as np
import pandas as pd
import time
import datetime
import random
used_ids = []
total_movements = 0
def get_grid(model):
    '''
    Esta es una función auxiliar que nos permite guardar el grid para cada uno de los agentes.
    param model: El modelo del cual optener el grid.
    return una matriz con la información del grid del agente.
    '''
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        if isinstance(cell_content, list):
            banqueta = next((agent for agent in cell_content if isinstance(agent, Banqueta)), None)
            auto = next((agent for agent in cell_content if isinstance(agent, Auto)), None)
            if banqueta:
                grid[pos[0]][pos[1]] = 1
            elif auto:
                grid[pos[0]][pos[1]] = 2
    return grid

class Auto(Agent):
    """An agent that moves towards two target coordinates."""
    def __init__(self, pos, model, unique_id, target1, target2):
        # Pass the parameters to the parent class.
        super().__init__(pos, model)
        self.wait_steps = 0
        # Create the agent's variables and set the initial values.
        self.target1 = target1
        self.target2 = target2
        self.at_target1 = False
        self.pos = pos
        print(self.pos)
        if pos == (31, 16):
          self.target1 = (15,16)
          self.target2 = (15,0)
        if pos == (31,17):
          self.target1 = (1,17)
          self.target2 = (0,17)
        if pos == (31,18):
          self.target1= (1, 18)
          self.target2= (0,18)
        if pos == (19,0):
          self.target1 = (19,12)
          self.target2 = (31,12)
        if pos == (16,0):
          self.target1 = (16,16)
          self.target2 = (0,16)
        if pos == (17,0):
          self.target1 = (17,1)
          self.target2 = (17,31)
        if pos == (18,0):
          self.target1 = (18,1)
          self.target2 = (18,31)
        if pos == (0,12):
          self.target1 = (12,12)
          self.target2 = (12,0)
        if pos == (0,15):
          self.target1 = (16,15)
          self.target2 = (16,31)
        if pos == (0,13):
          self.target1 = (1,13)
          self.target2 = (31,13)
        if pos == (0,14):
          self.target1 = (1,14)
          self.target2 = (31,14)
        if pos == (12,31):
          self.target1 = (12,19)
          self.target2 = (0,19)
        if pos == (15,31):
          self.target1 = (15,15)
          self.target2 = (31,15)
        if pos == (13,31):
          self.target1 = (13,30)
          self.target2 = (13,0)
        if pos == (14,31):
          self.target1 = (14,30)
          self.target2 = (14,0)

        global used_ids
        random_id = random.randint(1, 1000000) # You can change the range of random numbers as you wish
        while random_id in used_ids:
            random_id = random.randint(1, 1000000) # Generate another random number until it is not in the used_ids list
        used_ids.append(random_id) # Add the random number to the used_ids list
        self.unique_id = random_id
    def step(self):
        # Check if the agent has reached the first target
        if self.pos is None:
          return
        if self.wait_steps > 0:
            self.wait_steps -= 2
            return

        
            
        if not self.at_target1:
            # Move towards the first target
            new_pos = self.move_towards(self.target1)
            # Check if there is another car agent two blocks away in the current direction
            x, y = self.pos
            nx, ny = new_pos
            if x != nx:
                next_pos = (nx + (nx - x), y)
            else:
                next_pos = (x, ny + (ny - y))
            if new_pos == self.target1:
                self.at_target1 = True
        else:
            # Move towards the second target
            new_pos = self.move_towards(self.target2)
            # Check if there is another car agent two blocks away in the current direction
            x, y = self.pos
            nx, ny = new_pos
            if x != nx:
                next_pos = (nx + (nx - x), y)
            else:
                next_pos = (x, ny + (ny - y))
            if new_pos == self.target2:
                self.model.grid.remove_agent(self)
                return
        if self.wait_steps > 0:
            self.wait_steps -= 1
            return
        
        

        # Check if next_pos is within the grid boundaries
        grid_width, grid_height = self.model.grid.width, self.model.grid.height
        if 0 <= next_pos[0] < grid_width and 0 <= next_pos[1] < grid_height:
            cellmates = self.model.grid.get_cell_list_contents([next_pos])
            other_agents = [obj for obj in cellmates if isinstance(obj, Auto)]
            if len(other_agents) == 0:
                # Update the agent's position
                self.model.grid.move_agent(self, new_pos)
                
        global total_movements
        total_movements += 1



    def move_towards(self, target):
        """Move the agent one step closer to the target."""
        x, y = self.pos
        tx, ty = target

        # Compute the new x coordinate
        if x < tx:
            x += 1
        elif x > tx:
            x -= 1

        # Compute the new y coordinate
        if y < ty:
            y += 1
        elif y > ty:
            y -= 1

        return (x, y)

class Banqueta(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
class CleanerModel(Model):
      """A model with some number of agents."""

      def __init__(self, width, height, N, max_time, agent_positions):
          self.num_agents = N
          self.num_banquetas = 94
          self.banquetas = []
          self.grid = MultiGrid(width, height, True)
          self.schedule = SimultaneousActivation(self)
          self.max_time = max_time
          self.start_time = time.time()
          random.shuffle(agent_positions)
          # Create Succ agents in cell (1,1) with unique IDs
          counter = 0
          for i in range(self.num_agents):
              x, y = agent_positions[i]
              a = Auto((x, y), self, counter, target1=(19, 19), target2=(19, 31))
              counter += 1
              self.grid.place_agent(a, (x, y))
              self.schedule.add(a)

          for i in range(self.num_banquetas):
              banqueta = Banqueta(i, self)
              if i < 12:
                x = 0 + i
                y = 11
                self.grid.place_agent(banqueta, (x, y))
                x = 20 + i
                y = 11
                self.grid.place_agent(banqueta, (x, y))
                x = 0 + i
                y = 20
                self.grid.place_agent(banqueta, (x, y))
                x = 20 + i
                y = 20
                self.grid.place_agent(banqueta, (x, y))
                x = 11
                y = 0 + i
                self.grid.place_agent(banqueta, (x, y))
                x = 11
                y = 0 + i
                self.grid.place_agent(banqueta, (x, y))
                x = 20
                y = 0 + i
                self.grid.place_agent(banqueta, (x, y))
                x = 11
                y = 20 + i
                self.grid.place_agent(banqueta, (x, y))
                x = 20
                y = 20 + i
                self.grid.place_agent(banqueta, (x, y))
                self.schedule.add(banqueta)
          # Define the data collector to get the grid at each step
          self.datacollector = DataCollector(
              model_reporters={"Grid": get_grid})

      def step(self):
          '''
          At each step, the data collector will collect the data defined and store the grid to be plotted later.
          '''
          # Check if the maximum execution time has been reached
          agent_positions=[(31,19), (31,16) , (31,17), (31,18), (19,0), (16,0), (17,0), (18,0),(12,31), (13,31) , (14,31), (15,31), (0,12), (0,13), (0,14), (0,15)]
          if time.time() - self.start_time > self.max_time:
              return
          if self.schedule.steps % 5 == 0:
            x, y = agent_positions[random.randint(0, 15)]
            a = Auto((x,y), self, counter, target1=(19, 19), target2=(19, 31))
            self.grid.place_agent(a, (x,y))
            self.schedule.add(a)

          self.datacollector.collect(self)
          self.schedule.step()
          
# Definimos el tamaño del Grid
GRID_SIZE = 32
# Definimos el número de generaciones a correr
duration = 20 # seconds
frame_rate = 5 # frames per second do not alter
num_frames = duration * frame_rate
num_agents = 6
max=1
agent_positions=[(31,19), (31,16) , (31,17), (31,18), (19,0), (16,0), (17,0), (18,0),(12,31), (13,31) , (14,31), (15,31), (0,12), (0,13), (0,14), (0,15)]
# Create an instance of CleanerModel with user-defined parameters
model = CleanerModel(GRID_SIZE, GRID_SIZE, num_agents, max, agent_positions)

# Record start time
start_time = time.time()

counter = 1

# Run simulation until maximum execution time is reached or all cells are clean
while True:
    model.step()
    counter += 1
    if time.time() - model.start_time > model.max_time:
        break
# Get data from simulation
all_grid = model.datacollector.get_model_vars_dataframe()

print("Total de movimientos realizados por todos los autos:", total_movements)

from matplotlib.colors import ListedColormap

# Create a custom colormap that maps 0 to white, 1 to black, and 2 to blue
cmap = ListedColormap(['white', 'black', 'blue'])

# Create the animation using the custom colormap
fig, axs = plt.subplots(figsize=(5,5))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=cmap)

def animate(i):
    # Check if index is out-of-bounds
    if i >= len(all_grid):
        # If index is out-of-bounds, use last row of all_grid
        i = len(all_grid) - 1
    patch.set_data(all_grid.iloc[i][0])

anim = animation.FuncAnimation(fig, animate, frames=num_frames)
anim
plt.show()