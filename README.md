# HS21-LuxAI-NotSoSimple_Agent

Repo for the final project in the main seminar "Introduction to Computer Science" in HS21 at the University of Lucerne.

As a final project, the authors will participate in the public Kaggle Challenge *"Lux AI "*. More information can be found here: [Lux AI](https://www.kaggle.com/c/lux-ai-2021)

## The challenge

*The Lux AI Challenge is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand.*

- Quote from the Kaggle-page of the challenge, 23.11.2021




## Instructions



### 1. Read the [Getting Started]( https://github.com/Lux-AI-Challenge/Lux-Design-2021#getting-started) section and install the challenge.



### 2. Get our agent form the [GitHub-repository](https://github.com/Bazzarillo/HS21-LuxAI-NotSoSimple_Agent) and take a look at it.



### 3. In order to run the code please make sure you have installed the following libraries:

- all the lux-packages (should be installed in with the 1. step)
- math
- random
- pandas
- numpy
- csv
- collections
- datetime


### 4. In order to run a game you have to type the following code into the command prompt:
(NOTE: make sure you have set the working directory correctly)


#### main agent vs. main agent (the logfiles look a little confusing because they are created twice for the same agent.)
npx lux-ai-2021 main.py main.py 


#### main agnet vs. advanced_1
npx lux-ai-2021 main.py other_agents\advanced_1\main.py


#### main agent vs. advanced_2
npx lux-ai-2021 main.py other_agents\advanced_2\main.py


etc.


Please note that the other agents are all test agents, which represent the different working steps/progresses. They are neither tidy nor uniformly coded. In a way they serve as a reference point for the main agent.



### 5. Check out the following folders:

- errorlogs 
- log_and_statsfiles
- replays (you can upload the json-file [here](https://2021vis.lux-ai.org/) in order to watch an actual game)
- the ml_logger.csv may be used in the future for regression models by randomly chanig the value of some core variables. The variables and game results are stored in here.
