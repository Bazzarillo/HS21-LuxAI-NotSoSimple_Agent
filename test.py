import pandas as pd
from random import randint

height = 24
width = 24

x_map = []
y_map = []
res_map = []

for x in range (height):
    for y in range(width):
        x_map.append(x)
        y_map.append(y)
        res_map.append(randint(0,1))

df = pd.DataFrame({"x_map":x_map,"y_map":y_map, "res_map":res_map})

print(df)

start_x = 0
start_y = 0

end_x = height
end_y = width

x_start = []
y_start = []
res_dens = []
mean_container = []

while start_y <= end_y-2:
start_x = 0
    while start_x <= end_x-2:
        for tile_x in range(start_x + 2):
            for tile_y in range(start_y + 2):
                mean_container.append(df[(df["x_map"]==tile_x) & (df["y_map"]==tile_y)])
        
        mean_val = mean(mean_container)
        x_start.append(start_x)
        y_start.append(start_y)
        res_dens.append(mean_val)
        
    start_x += 1
start_y += 1







