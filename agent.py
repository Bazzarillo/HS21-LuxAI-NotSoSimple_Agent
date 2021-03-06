
# The following code is based on: https://youtu.be/6_GXTbTL9Uc
# You can find the whole code here: https://gist.github.com/Sentdex/971fa96b6b706627d69ba0a09ae437cc


# import libraries
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import os
import sys
import math
import random
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from collections import deque
from datetime import datetime
import csv
from functions import *


# specify variables
DIRECTIONS = Constants.DIRECTIONS
game_state = None
build_location = {}
ml_logger = csv.reader("ml_logger_lux.csv")

# create dictonaries needed to store relevant information
unit_to_city_dict = {}
unit_to_resource_dict = {}
worker_positions = {}
worker_task = {}
worker_goal = {}


# A task describes the behavioir of a worker.
tasks = [
    "Explorer", # Builds Cities
    "Mantainer" # "Ventures out into the wild" and gathers resources
    ]


# A goal describes the current goal of a worker.
goals = [
    "Build City",
    "Drop Resources"
]


# create log-file
now = datetime.now()
day = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H_%M_%S")
logfile = os.path.join("log_and_statsfiles", "agent_LOG_" + day + "_" + current_time + ".log")

# create statsfile (captures number of city tiles at the end of the game)
statsfile = os.path.join("log_and_statsfiles", "agent_STATS_" + day + "_" + current_time + ".txt")

######################## Actual AI-Code starts here ##########################


def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_city_dict
    global unit_to_resource_dict
    global worker_positions
    global max_cell
    global worker_per_city
    global fuel_constant
    global city_weight

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    # define game settings
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    resource_tiles = get_resource_tiles(game_state, width, height)
    workers = [u for u in player.units if u.is_worker()]



    ######################## Game constants that could be optimized ##########################


    ### Ratio of cities over workers

    # Currently, this value is held constant at 3/4.
    # We could optimize it by trying out random floats between 0.45 and 0.95
    # and see what works best. Uncomment code below to test.

    #worker_per_city = round(random.uniform(.45, .95),2)
    worker_per_city = 0.7

    ### Ratio of city fuel

    # Currently held constant at 300. 
    # Every worker is assigned to a city of n city tiles. Each city tile should at least have a resource
    # reserve of 300. This could be further optimized. Uncomment the code below to test.

    #fuel_constant = randint(50, 400)
    fuel_constant = 250

    
    
    ############################################# Setup #############################################


    ## 1)

    for unit in player.units:
        # if roughly 70% of the playtime has passed and a daycicle starts: search for new building spots.
        if observation["step"] == 239:
            # Get the cell with the highest resource density
            max_cell = get_resource_density(game_state, height, width, observation, unit, resource_tiles, player)
            with open(logfile, "a") as f:
                f.write(f"{observation['step']:}: Defining Max Density area: {max_cell}\n\n")


    ## 2)

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

        # capture workers' postions
        if w.id in worker_positions:
            worker_positions[w.id].append((w.pos.x, w.pos.y))
        else:
            # only log the last 3 rounds
            worker_positions[w.id] = deque(maxlen=3)
            worker_positions[w.id].append((w.pos.x, w.pos.y))

        # Make sure that only Mantainer get assigned to a city.
        # The first worker needs to be assigned to a city as the city otherwise dies out.
        if w.id not in unit_to_city_dict:
            with open(logfile, "a") as f:
                f.write(f"{observation['step']:}: Found mantainer unaccounted for {w.id}\n\n")
            city_assignment = get_close_city(player, w)
            unit_to_city_dict[w.id] = city_assignment

    with open(logfile, "a") as f:
        f.write(f"{observation['step']:}: Worker Positions {worker_positions}\n\n")

    # The first explorer should directly go to the max density area
    for w in workers:
        if w.id not in unit_to_resource_dict:
            with open(logfile, "a") as f:
                f.write(f"{observation['step']:}: Found worker w/o resource {w.id}\n\n")
            resource_assignment = get_close_resource(w, resource_tiles, player)
            unit_to_resource_dict[w.id] = resource_assignment


    
    ## 3)

    cities = player.cities.values()
    city_tiles = []

    # create list with city tiles
    for city in cities:
        for c_tile in city.citytiles:
            city_tiles.append(c_tile)

    # conditions need to be met before building a city...
    build_city = False

    try:
        """
        What is a good worker to city-tiles-ratio?
        Assumption: 3/4 => more city tiles is better...
        """
        if len(workers) / len(city_tiles) >= worker_per_city:
            build_city = True
        
    except:
        build_city = True



    ############################################# Action #############################################
    
    # We iterate over all our units and do something with them
    for unit in player.units:

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
                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']:}: Looks like a stuck worker {unit.id} - {last_positions}\n\n")
                        
                        # collision-solver ("random walk")
                        actions.append(unit.move(random.choice(["n","s","e","w"])))
                        continue

                # all workers with cargo space left are assigned to mine 
                if unit.get_cargo_space_left() > 0:

                    # where is the unit going?
                    intended_resource = unit_to_resource_dict[unit.id]
                    cell = game_state.map.get_cell(intended_resource.pos.x, intended_resource.pos.y)

                    # does the intended cell have resources?
                    # yes => move towards the cell
                    if cell.has_resource():
                        actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))
                    
                    # no => find another one and move towards this new cell
                    else:
                        intended_resource = get_close_resource(unit, resource_tiles, player)
                        unit_to_resource_dict[unit.id] = intended_resource
                        actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))

                # some workers should build cities!
                else:
                    # Max Explorers should start to build as soon as possible
                    # build_city = true if the worker-city_tile-ratio is 0.75
                    if build_city == True:

                        try:
                            # to which city is the worker assigned to?
                            associated_city_id = unit_to_city_dict[unit.id].cityid
                            # take the first city
                            unit_city = [c for c in cities if c.cityid == associated_city_id][0]
                            unit_city_fuel = unit_city.fuel
                            unit_city_size = len(unit_city.citytiles)
                            
                            # we need some fuel to survive the night, but how much?
                            # assumption: ~300 
                            enough_fuel = (unit_city_fuel/unit_city_size) > fuel_constant
                        
                        except: continue

                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']:}: Stuff needed for building a City ({associated_city_id}) fuel: {unit_city_fuel}, size: {unit_city_size}, enough fuel: {enough_fuel}\n\n")

                        # if we have enough fuel, we can try to build another city
                        if enough_fuel  and worker_goal[unit.id] != "Drop Ressources":
                            
                            with open(logfile, "a") as f:
                                f.write(f"{observation['step']:}: We WANT to build a city!\n\n")
                            
                            # but where do we want to build it?
                            # if we do not have a build location yet...
                            if unit.id not in build_location:
                                build_location[unit.id] = None

                            # ... we need to find one
                            if build_location[unit.id] is None:
                                # at the beginning: near to other cities
                                # later: build cities where the resource-density is high
                                # not every unit should go further away...
                                if  observation["step"] >= 240:
                                    if worker_task[unit.id] == "Explorer" or worker_task[unit.id] == "Max_Explorer":
                                        max_res = get_first_resource_max(max_cell, game_state)
                                        near_what = [max_res.pos.x, max_res.pos.y]
                                        build_location[unit.id] = find_empty_tile_near_2(near_what, game_state, observation)
                                        with open(logfile, "a") as f:
                                            f.write(f"{observation['step']:}: Building City in max area {max_cell}\n\n")
                                    elif len(player.cities) == 0:
                                        near_what = unit.pos
                                    else:
                                        near_what = get_close_city(player, unit)

                                else:
                                    # We mostly want to build next to other cities, but spread out to resources.
                                    loc_list = ["City","Resource"]
                                    city_weight = 6
                                    resource_weight = 10 - city_weight
                                    near = random.choices(loc_list,weights = [city_weight,resource_weight], k = 1)

                                    if len(player.cities) == 0:
                                        near_what = unit.pos
                                    elif near == "Resource":
                                        near_what = get_close_resource(unit, resource_tiles, player)
                                    else:
                                        near_what = get_close_city(player, unit)
                                    
                                    build_location[unit.id] = find_empty_tile_near_1(near_what, game_state, observation)
                                    with open(logfile, "a") as f:
                                        f.write(f"{observation['step']:}: Building City around standard area {near_what}\n\n")

                            # If the unit is already on a build location 
                            # => build!
                            if unit.pos == build_location[unit.id].pos:
                                action = unit.build_city()
                                actions.append(action)
                                
                                # Change Explorer to maintainer of it's city
                                if worker_task[unit.id] == "Explorer" or worker_task[unit.id] == "Max_Explorer":
                                    worker_task[unit.id] = "Mantainer"
                                    unit_to_city_dict[unit.id] = get_close_city(player,unit)

                                # reset variables if city is built!
                                build_city = False
                                build_location[unit.id] = None
                                
                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']:}: ### We BUILT the city! ###\n        Number of City Tiles: {len(city_tiles)}\n\n")
                                
                                continue   

                            # If the unit is not on a build location
                            else:
                                worker_goal[unit.id] = "Build City"

                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']:}: Navigating to where we wish to build!\n\n")
                                
                                # Navigating to where we wish to build a City
                                # actions.append(unit.move(unit.pos.direction_to(build_location.pos)))
                                dir_diff = (build_location[unit.id].pos.x - unit.pos.x, build_location[unit.id].pos.y - unit.pos.y)
                                xdiff = dir_diff[0]
                                ydiff = dir_diff[1]

                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']:}: dir_diff: {dir_diff} xdiff: {dir_diff[0]} ydiff: {dir_diff[1]}\n\n")

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
                                    f.write(f"{observation['step']:}: ### Actions: {actions}\n\n")

                                continue

                        elif len(player.cities) > 0:
                            worker_goal[unit.id] = "Drop Resources"

                            if unit.id in unit_to_city_dict and unit_to_city_dict[unit.id] in city_tiles:
                                move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                                actions.append(unit.move(move_dir))

                            else:
                                unit_to_city_dict[unit.id] = get_close_city(player,unit)
                                move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                                actions.append(unit.move(move_dir))

                    # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                    else:
                        worker_goal[unit.id] = "Drop Resources"

                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']:}: We want to drop of resources\n\n")

                        if unit.id in unit_to_city_dict and unit_to_city_dict[unit.id] in city_tiles:
                            # TODO: Test and delete.
                            unit_to_city_dict[unit.id] = get_close_city(player,unit)
                            move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                            actions.append(unit.move(move_dir))
                            with open(logfile, "a") as f:
                                f.write(f"{observation['step']:}: We need to get closer to the city\n\n")

                        # if appointed city died out, assign an other
                        else:
                            unit_to_city_dict[unit.id] = get_close_city(player,unit)
                            move_dir = unit.pos.direction_to(unit_to_city_dict[unit.id].pos)
                            actions.append(unit.move(move_dir))

                # if there are no city tiles left, build a new one

                if len(city_tiles) == 0:
                    if unit.get_cargo_space_left() == 0:
                        action = unit.build_city()
                        actions.append(action)
                        worker_task[unit.id] = "Mantainer"
                        unit_to_city_dict[unit.id] = get_close_city(player,unit)
            
            except Exception as e:
                with open(logfile, "a") as f:
                    f.write(f"{observation['step']:}: Unit error {str(e)}\n\n")

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
                        f.write(f"{observation['step']:}: Created a worker!\n\n")

                # if we cannot create a worker: => research!
                # NOTE: more than 200 research points is a waste of resources!
                else:
                    if player.research_points <= 199:
                        actions.append(city_tile.research())
                        
                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']:}: Doing research!\n\n")     

    if observation["step"] == 359:
        cities = player.cities.values()
        total_fuel = 0
        
        for city in cities:
            fuel = city.fuel
            total_fuel += fuel

        # capture number of CityTiles at the end of the game
        with open(statsfile,"a") as f:
            f.write("\n\n################################# GAME STATS #################################\n\n\n\n")
            f.write(f"### Map stats\n\n### Width: {width}\n\n### Length: {width}\n\n\n\n")
            f.write(f"### Number of City Tiles: {len(city_tiles)}\n\n### Numer of Workers: {len(workers)}\n\n### Research Points: {player.research_points}\n\n\n\n")
        with open("ml_logger_lux.csv","a") as fd:
            #ID,Workers,Cities,Ressources_Total,Workers_per_City,Ressources_per_City,City_weight
            fd.write(f"{random.randint(0,1000000)},{len(workers)},{len(city_tiles)},{total_fuel},{worker_per_city},{fuel_constant},{city_weight}\n")

    return actions