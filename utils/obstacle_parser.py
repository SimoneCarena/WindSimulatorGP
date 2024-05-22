import json

from modules.Obstacle import Obstacle

def parse_obstacles(obstacles_data):
    obstacles = []
    for obstacle in obstacles_data:
        x = obstacle["x"]
        y = obstacle["y"]
        r = obstacle["r"]
        obstacles.append(Obstacle(x,y,r))

    return obstacles