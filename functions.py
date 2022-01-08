
######################### FUNCTIONS #########################

import math
from os import sep
import sys

from pandas.core.frame import DataFrame
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import numpy as np
from collections import deque
import random
from datetime import datetime
import pandas as pd
from functions import *


# specify variables
DIRECTIONS = Constants.DIRECTIONS
game_state = None
build_location = None

# create dictonaries needed to store relevant information
unit_to_city_dict = {}
unit_to_resource_dict = {}
worker_positions = {}
worker_task = {}
goals = {}



# create log-file
now = datetime.now()
day = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H_%M_%S")
logfile = "log_and_statsfiles\\agent_LOG_" + day + "_" + current_time + ".log"


## 1)
def get_resource_tiles(game_state, width, height):
    '''
    Recreating the get_ressources()-function to declutter the agent
    It creates a list of cells out of the coordinates of every ressource tile.
    ''' 
    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles


## 2)
def get_ressource_types(game_state, height, width):
    '''
    Seperate approach. Will be needed to create a df containing
    the type of a resource as well.
    '''
    full_df = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                step = list(str(x) + "," + str(y) + "," + str(cell.resource.type) + "," + str(cell.resource.amount))
                full_df.append(step)
    return full_df


## 3)
def get_resource_density(game_state, height, width, observation, unit, resource_tiles, player):
    '''
    Create a numpy-array that is 01-coded, 1 = resource and 0 = no resource.
    Convolute over the whole array.
    Get the 3x3-area with the highest resource density
    send the worker tho this area in the first round.
    '''
    #Generate the array    
    zero_array = np.zeros((height*width))

    zero_array = zero_array.reshape(height,width)

    #01 code the array 
    for x in range (height):
        for y in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                zero_array[x][y] = 1
            else:
                zero_array[x][y] = 0

    # Pooling (condensing the grid by a density value by looping over the whole grid and calculating the mean
    # resource density in this area)
    mat = zero_array  

    M, N = mat.shape
    # K & L are the pooling filter sizes.

    # take a filter size of 3 by 3 if the playfield with is 12, otherwise 4 by 4
    if width == 12:
        c = 3
    else:
        c = 4

    K = c
    L = c

    MK = M // K
    NL = N // L

    # I stole this list comprehension from stack overflow
    # searches for the mean in the filter area
    sol = mat[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))

    try:
        # find max resource
        max = np.where(sol == sol.max())
        # select first one (if there are multiple max tiles)
        max = max[0]
        # Mutliply by c as max is a pooled array
        max_coordinate = [int(max[0])*c,int(max[1])*c]

    # if an error occurs, search for a close resource.
    except:
        max_coordinate = get_close_resource(unit,resource_tiles, player)

    with open(logfile,"a") as f:
                    f.write(f"{observation['step']}: ### Found a hight density position:{max_coordinate} ###\n\n")
    return max_coordinate


## 4)
def get_first_resource_max(max_coordinate, game_state):
    '''
    This function gets the first resource tile in the area with the highest resource density.
    Used to send Workers to this direction.
    '''
    has_res = False
    while has_res == False:
        for y in range(4):
            cell = game_state.map.get_cell(max_coordinate[0],max_coordinate[1]+y)
            if cell.has_resource() != None:
                goal = cell
                has_res = True
            for x in range(4):
                cell = game_state.map.get_cell(max_coordinate[0]+x,max_coordinate[1]+y)
                if cell.has_resource() != None:
                    goal = cell
                    has_res = True
    
    return goal


## 5)

def get_close_resource(unit, resource_tiles, player):
    """
    finds close resource tiles
    """
    closest_dist = math.inf
    closest_resource_tile = None
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:
        # Check if ressource tile is coal and if uranium is already researched.
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        # Check if ressource tile is uranium and if uranium is already researched.
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        # This dict contains a mapping of every active Unit. It gets assigned a ressource tile which it is going to harvest.
        if resource_tile in unit_to_resource_dict.values(): continue    

        dist = resource_tile.pos.distance_to(unit.pos)
        # This chunck of code takes the first value (form "infinite", to finite) and then optimizes for every resouce_tile in the list
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


## 6)
def get_close_city(player, unit):
    """
    finds close city tiles
    """
    closest_dist = math.inf
    closest_city_tile = None
    
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile
    return closest_city_tile


## 7)
def find_empty_tile_near_1(near_what, game_state, observation):
    """
    a 5x5 squared area (near_what in the center) gets scanned...
    """    
    build_location = None
    
    # set the search-scope (5x5-square)
    x_coord = range(-2, 3)
    y_coord = range(-2, 3)

    # get all possible combinations
    dirs = []
    for y in y_coord:
         for x in x_coord:
             dirs.append((x, y))

    # remove start position and sort        
    dirs.remove((0, 0))
    dirs = sorted(dirs)


    # sort the list again (it does not make any sense to start with tiles too far away...)
    # define indices  we want at the first positions (has to be set manually...)
    index=[7, 11, 16, 12, 6, 15, 17, 8, 2, 10, 21, 13, 1, 14, 22, 9, 3, 5, 20, 18, 0, 19, 23, 4]

    # apply order to the list
    dirs = [dirs[ind] for ind in index]

    # iterate through the list
    for d in dirs:
        try:
          possible_empty_tile = game_state.map.get_cell(near_what.pos.x+d[0], near_what.pos.y+d[1])
        
          if possible_empty_tile.resource == None and possible_empty_tile.road == 0 and possible_empty_tile.citytile == None:
             build_location = possible_empty_tile
            
             # logging
             with open(logfile,"a") as f:
                 f.write(f"{observation['step']}: Found build location:{build_location.pos}\n\n")
             return build_location
    
        # catch errors
        except Exception as e:
         with open(logfile,"a") as f:
             f.write(f"{observation['step']}: While searching for empty tiles:{str(e)}\n\n")

    with open(logfile,"a") as f:
     f.write(f"{observation['step']}: Something likely went wrong, couldn't find any empty tile\n\n")
     
    return None




## 8)
def find_empty_tile_near_2(near_what, game_state, observation):
    """
    optimize the function => now a 5x5 squared area (unit in the center) gets scanned...
    """
    build_location = None
    
    # set the search-scope (5x5-square)
    x_coord = range(-2, 3)
    y_coord = range(-2, 3)

    # get all possible combinations
    dirs = []
    for y in y_coord:
         for x in x_coord:
             dirs.append((x, y))

    # remove start position and sort         
    dirs.remove((0, 0))
    dirs = sorted(dirs)


    # sort the list again (it does not make any sense to start with tiles too far away...)
    # define indices  we want at the first positions (has to be set manually...)
    index=[7, 11, 16, 12, 6, 15, 17, 8, 2, 10, 21, 13, 1, 14, 22, 9, 3, 5, 20, 18, 0, 19, 23, 4]

    # apply order to the list
    dirs = [dirs[ind] for ind in index]

    # iterate through the list
    for d in dirs:
        try:
          possible_empty_tile = game_state.map.get_cell(near_what[0]+d[0], near_what[1]+d[1])
        
          if possible_empty_tile.resource == None and possible_empty_tile.road == 0 and possible_empty_tile.citytile == None:
             build_location = possible_empty_tile
            
             # logging
             with open(logfile,"a") as f:
                 f.write(f"{observation['step']}: Found build location:{build_location.pos}\n\n")
             return build_location
    
        # catch errors
        except Exception as e:
         with open(logfile,"a") as f:
             f.write(f"{observation['step']}: While searching for empty tiles:{str(e)}\n\n")

    with open(logfile,"a") as f:
     f.write(f"{observation['step']}: Something likely went wrong, couldn't find any empty tile\n\n")
     
    return None