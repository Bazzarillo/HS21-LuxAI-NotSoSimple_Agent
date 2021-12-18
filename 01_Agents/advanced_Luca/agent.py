import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import numpy as np
from collections import deque
import random
from datetime import datetime


# specify variables
DIRECTIONS = Constants.DIRECTIONS
game_state = None
build_location = None

# create dictonaries needed to store relevant information
unit_to_city_dict = {}
unit_to_resource_dict = {}
worker_positions = {}

# create log-file
now = datetime.now()
day = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H_%M_%S")
logfile = "agent_Luca_" + day + "_" + current_time + ".log"

# create statsfile (captures number of city tiles at the end of the game)
statsfile = "agent_stats_Luca_" + day + "_" + current_time + ".txt"





######################## define funtions needed below ########################

## 1) get_resource_tiles
# creates a list of cells (x- and y-coordinates) with resources
def get_resource_tiles(game_state, width, height):
    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    return resource_tiles


## 2) get_close_resource
def get_close_resource(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_resource_tile = None
    
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:

        # check the resource tile's type and if the resource is already researched or not
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue

        # add resource_tile to the unit_to_resource_dict
        if resource_tile in unit_to_resource_dict.values(): continue    

        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    
    return closest_resource_tile


## 3) get_close_city
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


## 4) find_empty_tile_near
## optimize the function => now a 5x5 squared area (unit in the center) gets scanned...
def find_empty_tile_near(near_what, game_state, observation):
    
    build_location = None
    
    # set the search-scope (5x5-square)
    x_coord = range(-2, 3)
    y_coord = range(-2, 3)

    # get all possible combinations
    dirs = []
    for y in y_coord:
         for x in x_coord:
             dirs.append((x, y))

    # remove position the worker is on and sort        
    dirs.remove((0, 0))
    dirs = sorted(dirs)


    # sort the list again (it does not make any sense to start with tiles too far away...)
    # define inidices we want at the first positions (has to be set manually...)
    index=[7, 11, 16, 12, 6, 15, 17, 8, 2, 10, 21, 13, 1, 14, 22, 9, 3, 5, 20, 18, 0, 19, 23, 4]

    # apply order to the list
    dirs = [dirs[ind] for ind in index]

    # split list into groups of 4
    n = 4
    dirs = [dirs[i:i+n] for i in range(0, len(dirs), n)] 

    # search for empty tiles
    for list_element in dirs:
        for d in list_element:
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



######################## Actual AI-Code starts here ##########################

def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_city_dict
    global unit_to_resource_dict
    global worker_positions

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



    ### Setup

    ## 1)
    for w in workers:

        # capture wokers' positions
        if w.id in worker_positions:
            worker_positions[w.id].append((w.pos.x, w.pos.y))
        else:
            # only log the last 3 rounds
            worker_positions[w.id] = deque(maxlen=3)
            worker_positions[w.id].append((w.pos.x, w.pos.y))

        
        # capture if workers are accounted to a city  
        if w.id not in unit_to_city_dict:
            
            with open(logfile, "a") as f:
                f.write(f"{observation['step']}: Found worker unaccounted for {w.id}\n\n")
            
            # assign them to a city
            city_assignment = get_close_city(player, w)
            unit_to_city_dict[w.id] = city_assignment

    # log wokers' positions
    with open(logfile, "a") as f:
        f.write(f"{observation['step']}: Worker Positions {worker_positions}\n\n")



    ## 2)
    for w in workers:

        # capture if workers have resources
        if w.id not in unit_to_resource_dict:
            with open(logfile, "a") as f:
                f.write(f"{observation['step']}: Found worker w/o resource {w.id}\n\n")

            # resource assignment
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
        # what is a good worker to city-tiles-ratio?
        # Assumption: 3/4 => more city tiles is better...
        if len(workers) / len(city_tiles) >= 0.75:
            build_city = True

    except:
        build_city = True



    ### Action
    # we iterate over all our units and do something with them
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
                # Sidenote: Wouldn't a division of the workers make sense?
                else:

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
                            enough_fuel = (unit_city_fuel/unit_city_size) > 300
                        
                        except: continue

                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']}: Stuff needed for building a City ({associated_city_id}) fuel: {unit_city_fuel}, size: {unit_city_size}, enough fuel: {enough_fuel}\n\n")

                        # if we have enough fuel, we can try to build another city
                        if enough_fuel:
                            
                            with open(logfile, "a") as f:
                                f.write(f"{observation['step']}: We want to build a city!\n\n")
                            
                            
                            # but where do we want to build it?
                            
                            # if we do not have a build location yet...
                            # find one near an existing one 
                            if build_location is None:
                                # near to other cities
                                # empty_near = get_close_city(player, unit)

                                # vs. near to resources
                                # maybe it would be better if the cities were built near to resources...
                                # technically it is better but we need to focus on resource-density instead of single resource-tiles...
                                # => use Mirco's funtion here!
                                
                                empty_near = get_close_resource(unit, resource_tiles, player)

                                build_location = find_empty_tile_near(empty_near, game_state, observation)

                            # If the unit is already on a build location 
                            # => build!
                            if unit.pos == build_location.pos:
                                action = unit.build_city()
                                actions.append(action)

                                # reset variables
                                build_city = False
                                build_location = None
                                
                                with open(logfile, "a") as f:
                                    f.write(f"{observation['step']}: ### We BUILT the city! ###\n        Number of City Tiles: {len(city_tiles)}\n\n")
                                
                                continue   

                            # If the unit is not on a build location
                            else:
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
                    if player.research_points <= 199:
                        actions.append(city_tile.research())
                        
                        with open(logfile, "a") as f:
                            f.write(f"{observation['step']}: Doing research! \n\n")

    if observation["step"] == 359:

        # capture number of CityTiles at the end of the game
        with open(statsfile,"a") as f:
            f.write("\n\n################################# GAME STATS #################################\n\n\n\n")
            f.write(f"### Number of City Tiles: {len(city_tiles)}\n\n### Numer of Workers: {len(workers)}\n\n### Research Points: {player.research_points}\n\n\n\n")
            f.write(f"### All units (list): {player.units}\n\n### Wokers (list): {workers}\n\n### city_tiles(list): {city_tiles}\n\n### resource_tiles(list): {resource_tiles}\n\n### resource_tiles[0]: {resource_tiles[0]}\n\n\n\n")
            f.write(f"### worker_positions(dict): {worker_positions}\n\n### unit_to_city_dict: {unit_to_city_dict}\n\n### unit_to_resource_dict: {unit_to_resource_dict}\n\n\n\n")
            f.write(f"### get_close_resource: {get_close_resource(unit, resource_tiles, player)}\n\n### get_close_city: {get_close_city(player, unit)}\n\n### build_location (variable): {build_location}\n\n### actions: {actions}\n\n\n\n")

    
    return actions

