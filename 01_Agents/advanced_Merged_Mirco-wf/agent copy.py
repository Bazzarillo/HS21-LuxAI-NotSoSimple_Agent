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
from functions import *
import csv


# specify variables
DIRECTIONS = Constants.DIRECTIONS
game_state = None
build_location = None
ml_logger = csv.reader("ml_logger_lux.csv")

# create dictonaries needed to store relevant information
unit_to_city_dict = {}
unit_to_resource_dict = {}
worker_positions = {}
worker_task = {}
worker_goal = {}
worker_goal_city = {}


# A task describes the behavioir of a worker.
tasks = [
    "Explorer", # Builds Cities
    "Mantainer" # "Ventures out into the wild" and gathers resources
    ]

# A goal describes the current goal of a worker.

goals = [
    "Build City",
    "Gather Resources",
    "Drop Resources"
]

# create log-file
now = datetime.now()
day = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H_%M_%S")
logfile = "agent_MERGED_" + day + "_" + current_time + ".log"

# create statsfile (captures number of city tiles at the end of the game)
statsfile = "agent_stats_MERGED_" + day + "_" + current_time + ".txt"

######################## Game constants that could be optimized ##########################

### Ratio of cities over workers

# Currently, this value is held constant at 3/4.
# We could optimize it by trying out random floats between 0.45 and 0.95
# and see what works best. uncomment code below to test.

#worker_per_city = round(random.uniform(.45, .95),2)
worker_per_city = 0.75

### Ratio of city fuel

# Currently held constant at 300. 
# Every worker is assigned to a city of n city tiles. Each city tile should at least have a resource
# reserve of 300. This could be further optimized. To do so, uncomment code below.

#fuel_constant = randint(50, 400)
fuel_constant = 300

######################## Actual AI-Code starts here ##########################

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

    ############################################# Setup #############################################

    ## 1)

    for w in workers:
        
        '''
        Split workers into Mantainers and Explorers.
        Explorers roam the map and bild new cities.
        Mantainers work on their city tiles.

        TODO: We shoul maybe optimize the assignment so that it's not 50/50
        '''

        # Make sure the first worker is Mantainer, otherwise the city dies out.
        if w.id in ["u_1", "u_2"]:
            worker_task[w.id] = "Mantainer"
        else:
            worker_task[w.id] = random.choice(tasks)
            if worker_task[w.id] == "Explorer" and "Max_Explorer" not in worker_task:
                worker_task[w.id] = "Max_Explorer"

        if w.id in worker_positions:
            worker_positions[w.id].append((w.pos.x, w.pos.y))
        else:
            worker_positions[w.id] = deque(maxlen=3)
            worker_positions[w.id].append((w.pos.x, w.pos.y))

        # Make sure that only Mantainer get assigned to a city.
        # The first worker needs to be assigned to a city as the city otherwise dies out.
        if w.id not in unit_to_city_dict:
            with open(logfile, "a") as f:
                f.write(f"{observation['step']} Found mantainer unaccounted for {w.id}\n")
            city_assignment = get_close_city(player, w)
            unit_to_city_dict[w.id] = city_assignment

    with open(logfile, "a") as f:
        f.write(f"{observation['step']} Worker Positions {worker_positions}\n")

    # Define the first explorer that directly goes to the max density area
    for w in workers:
        if w.id not in unit_to_resource_dict and worker_task[w.id] == "Max_Explorer":
            with open(logfile, "a") as f:
                f.write(f"{observation['step']} Found the Max-Density Explorer {w.id}\n")
            resource_assignment = get_first_resource_max(max_cell, game_state)
            unit_to_resource_dict[w.id] = resource_assignment
        else:
            if w.id not in unit_to_resource_dict:
                with open(logfile, "a") as f:
                    f.write(f"{observation['step']} Found worker w/o resource {w.id}\n")
                resource_assignment = get_close_resource(w, resource_tiles, player)
                unit_to_resource_dict[w.id] = resource_assignment

    ## 6)
    cities = player.cities.values()
    city_tiles = []

    # create list with city tiles
    for city in cities:
        for c_tile in city.citytiles:
            city_tiles.append(c_tile)

    # conditions need to be met before building a city...
    build_city = False

    try:
        # what is a good worker to city-tiles-ratio?
        # Assumption: 3/4 => more city tiles is better...
        # TODO: Optimize Ratio
        if len(workers) / len(city_tiles) >= worker_per_city:
            build_city = True

    except:
        build_city = True

    ############################################# Action #############################################
    
    # We iterate over all our units and do something with them
    for unit in player.units:
        #We need to update the max_cell as resources get deprivated over time
        if observation["step"] in [0,99,200]:
            # Get Cell in high resource density area
            max_cell = get_resource_density(game_state, height, width, observation, unit, resource_tiles, player)

        ## workers
        if unit.is_worker() and unit.can_act():
            
            try:
                last_positions = worker_positions[unit.id]
                
                # if the worker does not move for >= 2 rounds
                # => worker is stuck...
                if len(last_positions) >=2:
                    # get rid of duplicates
                    hm_positions = set(last_positions)
                    
                    # if the list is composed of only 1 pair of coordinates 
                    # => worker is stuck!
                    if len(list(hm_positions)) == 1:
                        
                        # logging
                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']}: Looks like a stuck worker {unit.id} - {last_positions}\n\n")
                        
                        # collision-solver ("random walk")
                        actions.append(unit.move(random.choice(["n","s","e","w"])))
                        continue

                # all workers with cargo space left are assigned to mine 
                if unit.get_cargo_space_left() > 0:

                    # where is the unit going?
                    intended_resource = unit_to_resource_dict[unit.id]
                    cell = game_state.map.get_cell(intended_resource.pos.x, intended_resource.pos.y)

                    # does the intended cell have resources?
                    # yes!
                    if cell.has_resource():
                        actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))
                    # no!
                    else:
                        intended_resource = get_close_resource(unit, resource_tiles, player)
                        unit_to_resource_dict[unit.id] = intended_resource
                        actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))

                # some workers should build cities!
                else:
                    
                    # Max Explorers should start to build as soon as possible
                    # build_city = true if the worker-city_tile-ratio is 0.75
                    if build_city:

                        try:
                            # to which city is the worker assigned to?
                            associated_city_id = unit_to_city_dict[unit.id].cityid
                            # take the first city
                            unit_city = [c for c in cities if c.cityid == associated_city_id][0]
                            unit_city_fuel = unit_city.fuel
                            unit_city_size = len(unit_city.citytiles)
                            
                            # we need some fuel to survive the night, but how much?
                            # assumption: ~300 
                            # TODO: Optimize Value
                            enough_fuel = (unit_city_fuel/unit_city_size) > fuel_constant
                        
                        except: continue

                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']}: Stuff needed for building a City ({associated_city_id}) fuel: {unit_city_fuel}, size: {unit_city_size}, enough fuel: {enough_fuel}\n\n")

                        # if we have enough fuel, we can try to build another city
                        if enough_fuel or worker_task[unit.id] == "Max_Explorer":
                            
                            with open(logfile, "a") as f:
                                f.write(f"{observation['step']}: We WANT to build a city!\n\n")
                            
                            ### but where do we want to build it?
                            
                            # if we do not have a build location yet...
                            if build_location is None:

                                # near to other cities
                                # later we should build cities where the resource-densitiy is high
                                # no every unit should go further away...
                                if  observation["step"] > 100:
                                    if worker_task[unit.id] == "Explorer" or worker_task[unit.id] == "Max_Explorer":
                                        near_what = max_cell
                                    else:
                                        near_what = get_close_city(player, unit)

                                else:
                                    near_what = get_close_city(player, unit)

                                # define build location
                                if near_what == max_cell:
                                    build_location = find_empty_tile_near_2(near_what, game_state, observation)
                                    with open(logfile, "a") as f:
                                        f.write(f"{observation['step']}: Building City in max area {max_cell} \n\n")
                                else:
                                    build_location = find_empty_tile_near_1(near_what, game_state, observation)
                                    with open(logfile, "a") as f:
                                        f.write(f"{observation['step']}: Building City around standard area {near_what} \n\n")

                            # If the unit is already on a build location 
                            # => build!
                            if unit.pos == build_location.pos:
                                action = unit.build_city()
                                actions.append(action)
                                worker_goal[unit.id] = "Gather Resources"
                                
                                if worker_task[unit.id] == "Explorer" or worker_task[unit.id] == "Max_Explorer":
                                    worker_task[unit.id] = "Mantainer"

                                # reset variables if city is built!
                                build_city = False
                                build_location = None
                                
                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']}: ### We BUILT the city! ###\n        Number of City Tiles: {len(city_tiles)}\n\n")
                                
                                continue   

                            # If the unit is not on a build location
                            else:
                                
                                worker_goal[unit.id] = "Build City"

                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']}: Navigating to where we wish to build!\n\n")
                                
                                # Navigating to where we wish to build a City
                                # actions.append(unit.move(unit.pos.direction_to(build_location.pos)))
                                dir_diff = (build_location.pos.x - unit.pos.x, build_location.pos.y - unit.pos.y)
                                xdiff = dir_diff[0]
                                ydiff = dir_diff[1]

                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']}: dir_diff: {dir_diff} xdiff: {dir_diff[0]} ydiff: {dir_diff[1]}\n\n")

                                # Where to go?
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
                                        # if there is a city tile 
                                        # => move in the other direction that we overall want to move
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
                                
                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']}: ### Actions: {actions}\n\n")

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
                    elif (unit.get_cargo_space_left() == 0 and worker_goal[unit.id] == "Mantainer") or worker_goal[unit.id] == "Drop Resources":
                    
                        worker_goal[unit.id] = "Drop Resources"

                        if unit.id in unit_to_city_dict and unit_to_city_dict[unit.id] in city_tiles:
                            
                            # If unit is already on city, change goal to gather resources
                            if unit.pos == unit_to_city_dict[unit.id].pos:
                                worker_goal[unit.id] = "Gather Resources"

                            else:
                                move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                                actions.append(unit.move(move_dir))

                        # if appointed city died out, assign an other
                        else:
                            unit_to_city_dict[unit.id] = get_close_city(player,unit)
                            move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                            actions.append(unit.move(move_dir))
            
            except Exception as e:
                with open(logfile, "a") as f:
                    f.write(f"{observation['step']}: Unit error {str(e)} \n\n")

    ## cities
    can_create = len(city_tiles) - len(workers)

    if len(city_tiles) > 0:
        for city_tile in city_tiles:

            # if cooldown is < 1 => unit can act
            if city_tile.can_act():
            
                # cities can only create workers if the # of workers + # of carts < # of city tiles
                # if ture => create a worker!
                if can_create > 0:
                    actions.append(city_tile.build_worker())
                    can_create -= 1
                    
                    with open(logfile, "a") as f:
                        f.write(f"{observation['step']}: Created a worker! \n\n")

                # if we cannot create a worker: => research!
                # NOTE: more than 200 research points is a waste of resources!
                else:
                    if player.research_points <= 200:
                        actions.append(city_tile.research())
                        
                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']}: Doing research! \n\n")     

    if observation["step"] == 359:

        # capture number of CityTiles at the end of the game
        with open(statsfile,"a") as f:
            f.write("\n\n################################# GAME STATS #################################\n\n\n\n")
            f.write(f"### Number of City Tiles: {len(city_tiles)}\n\n### Numer of Workers: {len(workers)}\n\n### Research Points: {player.research_points}\n\n\n\n")
            f.write(f"### Map stats\n\n### Width: {width}\n\n### Length: {width}\n\n\n\n")
            f.write(f"### All units (list): {player.units}\n\n### Wokers (list): {workers}\n\n### city_tiles(list): {city_tiles}\n\n### resource_tiles(list): {resource_tiles}\n\n### resource_tiles[0]: {resource_tiles[0]}\n\n\n\n")
            f.write(f"### worker_positions(dict): {worker_positions}\n\n### unit_to_city_dict: {unit_to_city_dict}\n\n### unit_to_resource_dict: {unit_to_resource_dict}\n\n\n\n")
            f.write(f"### get_close_resource: {get_close_resource(unit, resource_tiles, player)}\n\n### get_close_city: {get_close_city(player, unit)}\n\n\n\n")
            f.write(f"### max_cell (variable): {max_cell}\n\n### build_location (variable): {build_location}\n\n### actions: {actions}\n\n\n\n")
            f.write(f"empty_near = get_close_resource(unit, resource_tiles, player): {get_close_resource(unit, resource_tiles, player)}")

    return actions

