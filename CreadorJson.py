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
import json

used_ids = []
total_movements = 0
total_cars = 0
total_final = 0

def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        if isinstance(cell_content, list):
            auto = next((agent for agent in cell_content if isinstance(agent, Auto)), None)
            if auto:
                grid[pos[0]][pos[1]] = 1
    return grid

class Auto(Agent):
    def __init__(self, pos, model, unique_id, target1, target2):
        super().__init__(pos, model)
        self.wait_steps = 0
        self.target1 = target1
        self.target2 = target2
        self.at_target1 = False
        self.pos = pos
        if pos == (31, 16):
          self.target1 = (15,16)
          self.target2 = (15,0)
        if pos == (31, 19):
          self.target1 = (19,19)
          self.target2 = (19,31)
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
        random_id = random.randint(1, 1000000)
        while random_id in used_ids:
            random_id = random.randint(1, 1000000)
        used_ids.append(random_id)
        self.unique_id = random_id
        
    def step(self):
        if self.pos is None:
          return
        if self.wait_steps > 0:
            self.wait_steps -= 2
            return
        global total_movements
        total_movements += 1
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
                #self.model.grid.remove_agent(self)
                global total_final
                total_final += 1
                return
        if self.wait_steps > 0:
            self.wait_steps -= 1
            return

        grid_width, grid_height = self.model.grid.width, self.model.grid.height
        if 0 <= next_pos[0] < grid_width and 0 <= next_pos[1] < grid_height:
            cellmates = self.model.grid.get_cell_list_contents([next_pos])
            other_agents = [obj for obj in cellmates if isinstance(obj, Auto)]
            if len(other_agents) == 0:
                self.model.grid.move_agent(self, new_pos)

    def move_towards(self, target):
        x, y = self.pos
        tx, ty = target

        if x < tx:
            x += 1
        elif x > tx:
            x -= 1

        if y < ty:
            y += 1
        elif y > ty:
            y -= 1

        return (x, y)

class CleanerModel(Model):
      def __init__(self, width, height, N, max_steps, agent_positions):
          self.num_agents = N
          self.num_banquetas = 94
          self.banquetas = []
          self.grid = MultiGrid(width, height, True)
          self.schedule = SimultaneousActivation(self)
          self.step_count = 0
          self.max_steps = max_steps
          self.start_time = time.time()
          random.shuffle(agent_positions)
          self.all_steps_data = []
          self.step_data = []
          counter = 0
          for i in range(self.num_agents):
              x, y = agent_positions[i]
              a = Auto((x, y), self, counter, target1=(19, 19), target2=(19, 31))
              counter += 1
              self.grid.place_agent(a, (x, y))
              self.schedule.add(a)
          self.datacollector = DataCollector(
              agent_reporters={
                  "x": lambda a: a.pos[0] if a.pos is not None else None,
                  "z": lambda a: a.pos[1] if a.pos is not None else None
              }
          )

      def step(self):
          agent_positions=[(31,19), (31,16) , (31,17), (31,18), (19,0), (16,0), (17,0), (18,0),(12,31), (13,31) , (14,31), (15,31), (0,12), (0,13), (0,14), (0,15)]
          if self.schedule.steps % 5 == 0:
            x, y = agent_positions[random.randint(0, 15)]
            a = Auto((x,y), self, counter, target1=(19, 19), target2=(19, 31))
            self.grid.place_agent(a, (x,y))
            self.schedule.add(a)
          print(self.step_count)
          print(self.max_steps)
          if self.step_count >= self.max_steps:
             print("Maximum number of steps reached!")
             return
          self.step_count += 1
          self.datacollector.collect(self)
          self.schedule.step()  
          return

GRID_SIZE = 32
duration = 20
frame_rate = 5
max_steps = 100
num_frames = max_steps
num_agents = 4
total_cars += num_agents
max=1
agent_positions=[(31,19), (31,16) , (31,17), (31,18), (19,0), (16,0), (17,0), (18,0),(12,31), (13,31) , (14,31), (15,31), (0,12), (0,13), (0,14), (0,15)]
model = CleanerModel(GRID_SIZE, GRID_SIZE, num_agents, max_steps, agent_positions)
start_time = time.time()
counter = 1
while True:
    model.step()
    counter += 1
    if model.step_count >= model.max_steps:
        break
file_path = "grid_data.txt"
data = model.datacollector.get_agent_vars_dataframe().reset_index().to_dict('records')
with open('modelo.json', 'w') as f:
    json.dump(data, f)