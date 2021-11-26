import math, sys

from pandas.core.frame import DataFrame
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import pandas as pd
import numpy as np
from collections import deque #for faster appending, see: https://www.geeksforgeeks.org/deque-in-python/
from datetime import datetime

# We need a Log that tells us what happened during the game
# Logs should not overwrite, thus we create a log per game.
now = datetime.now()

day = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H_%M_%S")

logfile = "agent_" + day + "_" + current_time + ".log"

open(logfile,"w")

DIRECTIONS = Constants.DIRECTIONS
game_state = None

unit_to_resource_dict = {}
worker_positions = {}
resource_frame = DataFrame()

statsfile = "agent.txt"

'''
Recreating the get_ressources()-function to declutter the agent

It creates a list of cells out of the coordinates of every ressource tile.
'''

def get_ressources(game_state, width, height):
    
    resource_tiles: list[Cell] = []
    for y in range(height):
            for x in range(width):
                cell = game_state.map.get_cell(x, y)
                if cell.has_resource():
                    resource_tiles.append(cell)
    return resource_tiles

'''
Seperate approach. Will be needed to create a df containing 
the type of a resource as well.
'''

def get_ressource_types(resource_tiles):
    resource_types = []

    for tile in resource_tiles:
        resource_types.append(tile.resource.type)

    resource_frame = pd.DataFrame(resource_tiles, columns = ["coor"])
    resource_frame["resource"] = resource_types
    return resource_frame

'''
Further decluttering basic agent agent code.
'''
def get_closest_resource(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
                        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
                        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
                        dist = resource_tile.pos.distance_to(unit.pos)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_resource_tile = resource_tile
    return closest_resource_tile

def agent(observation, configuration):
    global game_state
    #global build_location
    global unit_to_resource_dict
    global worker_positions
    global resource_frame

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height


    resource_tiles = get_ressources(game_state, width, height)
    resource_frame = get_ressource_types(resource_tiles)

    # Creates a list of workers
    workers = [u for u in player.units if u.is_worker()]

    for w in workers:

        if w.id in worker_positions:
            worker_positions[w.id].append((w.pos.x, w.pos.y))
        else:
            #TODO: is deque really needed? If still works while being commented
            #out: delete

            worker_positions[w.id] = deque(maxlen=3)
            worker_positions[w.id].append((w.pos.x, w.pos.y))

    # Add log for worker position
    with open(logfile, "a") as f:
        f.write(f"{observation['step']} Worker Positions {worker_positions}\n")
    
    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_worker() and unit.can_act():

            closest_resource_tile = get_closest_resource(unit,resource_tiles, player)

            if closest_resource_tile is not None:
                    actions.append(unit.move(unit.pos.direction_to(closest_resource_tile.pos)))
            else:
                # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                if len(player.cities) > 0:
                    closest_dist = math.inf
                    closest_city_tile = None
                    for k, city in player.cities.items():
                        for city_tile in city.citytiles:
                            dist = city_tile.pos.distance_to(unit.pos)
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_city_tile = city_tile
                    if closest_city_tile is not None:
                        move_dir = unit.pos.direction_to(closest_city_tile.pos)
                        actions.append(unit.move(move_dir))

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))

    return actions

resource_frame.to_csv("resource_frame.csv")
