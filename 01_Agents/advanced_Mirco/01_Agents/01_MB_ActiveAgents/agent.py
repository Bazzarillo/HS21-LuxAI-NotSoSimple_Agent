# Code associated w/: https://youtu.be/6_GXTbTL9Uc
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

now = datetime.now()

day = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H_%M_%S")

logfile = "agent_" + day + "_" + current_time + ".log"


open(logfile, "w")

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

def get_resource_density(game_state, height, width, observation):
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

    # Convolution (condensing the grid by a density value by looping over the whole grid and calculating the mean
    # resource density in this area)
    mat = zero_array  

    M, N = mat.shape
    # K & L are the convolution filter sizes.
    if width == 12:
        c = 3
    else:
        c = 4
    K = c
    L = c

    MK = M // K
    NL = N // L

    #I stole this list comprehension from stack overflow
    sol = mat[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))

    max = np.where(sol == sol.max())
    max = max[0]
    max_coordinate = [int(max[0])*c,int(max[1])*c]

    with open(logfile,"a") as f:
                    f.write(f"{observation['step']}: Found a hight density position:{max_coordinate}\n")
    return max_coordinate

def get_first_resource_max(max_coordinate):
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

def get_close_resource(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_resource_tile = None
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:
        # Check if ressource tile is coal and if uranium is already researched.
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        # Check if ressource tile is uranium and if uranium is already researched.
        # TODO: Implement ressource optimization - Always carry 80% wood, 15% Coal, 5% Uraniuum
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        # This dict contains a mapping of every active Unit. It gets assigned a ressource tile which it is going to harvest.
        if resource_tile in unit_to_resource_dict.values(): continue    

        dist = resource_tile.pos.distance_to(unit.pos)
        # This chunck of code takes the first value (form "infinite", to finite) and then optimizes for every resouce_tile in the list
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile



def get_close_city(player, unit):
    closest_dist = math.inf
    closest_city_tile = None
    
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile
    return closest_city_tile


def find_empty_tile_near(near_what, game_state, observation):
    build_location = None
    dirs = [(1,0), (0,1), (-1,0), (0,-1)]
    # may later need to try: dirs = [(1,-1), (-1,1), (-1,-1), (1,1)] too.
    for d in dirs:
        try:
            possible_empty_tile = game_state.map.get_cell(near_what.pos.x+d[0], near_what.pos.y+d[1])
            # logging.INFO(f"{observation['step']}: Checking:{possible_empty_tile.pos}")
            if possible_empty_tile.resource == None and possible_empty_tile.road == 0 and possible_empty_tile.citytile == None:
                build_location = possible_empty_tile
                with open(logfile,"a") as f:
                    f.write(f"{observation['step']}: Found build location:{build_location.pos}\n")

                return build_location
        except Exception as e:
            with open(logfile,"a") as f:
                f.write(f"{observation['step']}: While searching for empty tiles:{str(e)}\n")


    with open(logfile,"a") as f:
        f.write(f"{observation['step']}: Couldn't find a tile next to, checking diagonals instead...\n")

    dirs = [(1,-1), (-1,1), (-1,-1), (1,1)] 
    # may later need to try: dirs = [(1,-1), (-1,1), (-1,-1), (1,1)] too.
    for d in dirs:
        try:
            possible_empty_tile = game_state.map.get_cell(near_what.pos.x+d[0], near_what.pos.y+d[1])
            # logging.INFO(f"{observation['step']}: Checking:{possible_empty_tile.pos}")
            if possible_empty_tile.resource == None and possible_empty_tile.road == 0 and possible_empty_tile.citytile == None:
                build_location = possible_empty_tile
                with open(logfile,"a") as f:
                    f.write(f"{observation['step']}: Found build location:{build_location.pos}\n")

                return build_location
        except Exception as e:
            with open(logfile,"a") as f:
                f.write(f"{observation['step']}: While searching for empty tiles:{str(e)}\n")


    # PROBABLY should continue our search out with something like dirs = [(2,0), (0,2), (-2,0), (0,-2)]...
    # and so on


    with open(logfile,"a") as f:
        f.write(f"{observation['step']}: Something likely went wrong, couldn't find any empty tile\n")
    return None

# Create Log that tells us what happened during the game
# Logs should not overwrite, thus we create a log per game.

DIRECTIONS = Constants.DIRECTIONS
game_state = None
build_location = None

unit_to_city_dict = {}
unit_to_resource_dict = {}
worker_positions = {}
worker_task = {}
goals = {}

statsfile = "agent.txt"

def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_city_dict
    global unit_to_resource_dict
    global worker_positions
    global max_cell

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
    resource_tiles = get_resource_tiles(game_state, width, height)
    workers = [u for u in player.units if u.is_worker()]

    if observation["step"] == 0:
        # Get Cell in high resource density area
        max_cell = get_resource_density(game_state, height, width, observation)

    for w in workers:
        
        '''
        Split workers into Mantainers and Explorers.
        Explorers roam the map and bild new cities.
        Mantainers work on their city tiles.

        TODO: We shoul maybe randomize the assignment so that it's not 50/50
        '''
        wid = w.id
        if (int(wid[2:]) % 2) == 0:
            worker_task[w.id] = "Explorer"
        else:
            worker_task[w.id] = "Mantainer"

        if w.id in worker_positions:
            worker_positions[w.id].append((w.pos.x, w.pos.y))
        else:
            worker_positions[w.id] = deque(maxlen=3)
            worker_positions[w.id].append((w.pos.x, w.pos.y))

        if w.id not in unit_to_city_dict and worker_task[w.id] == "Mantainer":
            with open(logfile, "a") as f:
                f.write(f"{observation['step']} Found worker unaccounted for {w.id}\n")
            city_assignment = get_close_city(player, w)
            unit_to_city_dict[w.id] = city_assignment

        if w.id not in unit_to_city_dict and w.id == "u_1":
            with open(logfile, "a") as f:
                f.write(f"{observation['step']} Found worker unaccounted for {w.id}\n")
            city_assignment = get_close_city(player, w)
            unit_to_city_dict[w.id] = city_assignment

    with open(logfile, "a") as f:
        f.write(f"{observation['step']} Worker Positions {worker_positions}\n")

    for w in workers:
        if w.id not in unit_to_resource_dict and w.id == "u_4":
            with open(logfile, "a") as f:
                f.write(f"{observation['step']} Found the Max-Density Explorer{w.id}\n")
            resource_assignment = get_first_resource_max(max_cell)
            unit_to_resource_dict[w.id] = resource_assignment
        else:
            if w.id not in unit_to_resource_dict:
                with open(logfile, "a") as f:
                    f.write(f"{observation['step']} Found worker w/o resource {w.id}\n")
                resource_assignment = get_close_resource(w, resource_tiles, player)
                unit_to_resource_dict[w.id] = resource_assignment



    cities = player.cities.values()
    city_tiles = []

    for city in cities:
        for c_tile in city.citytiles:
            city_tiles.append(c_tile)
     

    build_city = False

    try:
        if len(workers) / len(city_tiles) >= 0.75:
            build_city = True
    except:
        build_city = True

    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            try:
                last_positions = worker_positions[unit.id]
                if len(last_positions) >=2:
                    hm_positions = set(last_positions)
                    if len(list(hm_positions)) == 1:
                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']} Looks like a stuck worker {unit.id} - {last_positions}\n")

                        actions.append(unit.move(random.choice(["n","s","e","w"])))
                        continue
          
                if unit.get_cargo_space_left() > 0:
                    intended_resource = unit_to_resource_dict[unit.id]
                    cell = game_state.map.get_cell(intended_resource.pos.x, intended_resource.pos.y)

                    if cell.has_resource():
                        actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))

                    else:
                        intended_resource = get_close_resource(unit, resource_tiles, player)
                        unit_to_resource_dict[unit.id] = intended_resource
                        actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))


                else:
                    if build_city:

                        try:
                            associated_city_id = unit_to_city_dict[unit.id].cityid
                            unit_city = [c for c in cities if c.cityid == associated_city_id][0]
                            unit_city_fuel = unit_city.fuel
                            unit_city_size = len(unit_city.citytiles)

                            enough_fuel = (unit_city_fuel/unit_city_size) > 300
                        except: continue

                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']} Build city stuff: {associated_city_id}, fuel {unit_city_fuel}, size {unit_city_size}, enough fuel {enough_fuel}\n")


                        if enough_fuel:
                            with open(logfile, "a") as f:
                                f.write(f"{observation['step']} We want to build a city!\n")
                            if build_location is None:
                                empty_near = get_close_city(player, unit)
                                build_location = find_empty_tile_near(empty_near, game_state, observation)


                            if unit.pos == build_location.pos:
                                action = unit.build_city()
                                actions.append(action)

                                build_city = False
                                build_location = None
                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']} Built the city!\n")
                                continue   

                            else:
                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']}: Navigating to where we wish to build!\n")

                                # actions.append(unit.move(unit.pos.direction_to(build_location.pos)))
                                dir_diff = (build_location.pos.x-unit.pos.x, build_location.pos.y-unit.pos.y)
                                xdiff = dir_diff[0]
                                ydiff = dir_diff[1]

                                # decrease in x? West
                                # increase in x? East
                                # decrease in y? North
                                # increase in y? South

                                if abs(ydiff) > abs(xdiff):
                                    # if the move is greater in the y axis, then lets consider moving once in that dir
                                    check_tile = game_state.map.get_cell(unit.pos.x, unit.pos.y+np.sign(ydiff))
                                    if check_tile.citytile == None:
                                        if np.sign(ydiff) == 1:
                                            actions.append(unit.move("s"))
                                        else:
                                            actions.append(unit.move("n"))

                                    else:
                                        # there's a city tile, so we want to move in the other direction that we overall want to move
                                        if np.sign(xdiff) == 1:
                                            actions.append(unit.move("e"))
                                        else:
                                            actions.append(unit.move("w"))

                                else:
                                    # if the move is greater in the y axis, then lets consider moving once in that dir
                                    check_tile = game_state.map.get_cell(unit.pos.x+np.sign(xdiff), unit.pos.y)
                                    if check_tile.citytile == None:
                                        if np.sign(xdiff) == 1:
                                            actions.append(unit.move("e"))
                                        else:
                                            actions.append(unit.move("w"))

                                    else:
                                        # there's a city tile, so we want to move in the other direction that we overall want to move
                                        if np.sign(ydiff) == 1:
                                            actions.append(unit.move("s"))
                                        else:
                                            actions.append(unit.move("n"))


                                continue

                        elif len(player.cities) > 0:
                            if unit.id in unit_to_city_dict and unit_to_city_dict[unit.id] in city_tiles:
                                move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                                actions.append(unit.move(move_dir))

                            else:
                                unit_to_city_dict[unit.id] = get_close_city(player,unit)
                                move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                                actions.append(unit.move(move_dir))




                    # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                    elif len(player.cities) > 0:
                        if unit.id in unit_to_city_dict and unit_to_city_dict[unit.id] in city_tiles:
                            move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                            actions.append(unit.move(move_dir))

                        else:
                            unit_to_city_dict[unit.id] = get_close_city(player,unit)
                            move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                            actions.append(unit.move(move_dir))
            except Exception as e:
                with open(logfile, "a") as f:
                    f.write(f"{observation['step']}: Unit error {str(e)} \n")



    can_create = len(city_tiles) - len(workers)

    if len(city_tiles) > 0:
        for city_tile in city_tiles:
            if city_tile.can_act():
                if can_create > 0:
                    actions.append(city_tile.build_worker())
                    can_create -= 1
                    with open(logfile, "a") as f:
                        f.write(f"{observation['step']}: Created and worker \n")
                else:
                    actions.append(city_tile.research())
                    with open(logfile, "a") as f:
                        f.write(f"{observation['step']}: Doing research! \n")


    if observation["step"] == 359:
        with open(statsfile,"a") as f:
            f.write(f"{len(city_tiles)}\n")
    
    return actions