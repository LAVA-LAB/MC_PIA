# COOL-MC

Note: This is the combined branch of adversarial_rl2 and rl_robustness. If any problems occur during adversarial RL or robustness checking, please refer to the original repositories. To reproduce the experiments in Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking, we refer to point 17 in the table of content.

COOL-MC provides a framework for connecting state-of-the-art (deep) reinforcement learning (RL) with modern model checking. In particular, COOL-MC extends the OpenAI Gym to support RL training on PRISM environments and allows verification of the trained RL policies via the Storm model checker.
The general workflow of our approach is as follows. First, we model the RL environment as a MDP in PRISM. Second, we train our RL policy in the PRISM environment or, if available, in the matching OpenAI Gym environment. Third, we verify the trained RL policy via the Storm model checker. Depending on the model checking outcome, we retrain the RL policy or deploy it. We are convinced that the basis provided by the tool helps those interested in connecting the areas of verification and RL with the proper framework to create new approaches in an effective and reproducible way.

Paper: [here](https://arxiv.org/abs/2209.07133)

![workflow](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/workflow_diagram.png)

The following diagram shows the major components of the tool and their interactions:


![components](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/use-case-diagram.png)

The *RL agent* is a wrapper around the trained policy and interacts with the environment. Currently implemented agents include Q-Learning, Hillclimbing, Deep Q-Learning, and REINFORCE.
From a training perspective, the RL agent can be trained via the *Storm simulator* or an *OpenAI Gym*.
From a verification perspective, the *model builder* uses the Storm
simulator to incrementally build a DTMC (see next diagram), which is then *model checked* by Storm.
Note that we only build the part of the model which is visited by the policy. 
For every state, the policy (the RL agent) is queried for an action. Now, according to the PRISM model,
all transitions and states that may be reached via that action are explored to
incrementally build a model that is restricted to the action-choices of the trained
policy. The resulting model is fully probabilistic, as no action choices are left
open, and is a Markov chain induced by the original MDP and the policy.


##### Content
1. Getting Started with COOL-MC
2. Example 1 (Frozen Lake)
3. Example 2 (Taxi)
4. Example 3 (Collision Avoidance)
5. Example 4 (Smart Grid)
6. Example 5 (Warehouse)
7. Example 6 (Stock Market)
8. Example 7 (James Bond 007)
9. Example 8 (Crazy Climber)
10. Benchmarks
11. Web Interface
12. Model Checking Times and Limitations
13. PRISM Modelling Tips
14. RL Agent Training
15. COOL-MC Command Line Arguments
16. Manual Installation
17. Reproducing the results of the paper Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking

## Getting Started with COOL-MC
We assume that you have docker installed and that you run the following commands in the root of this repository:
1. Download the docker container [here](https://www.dropbox.com/s/ovusoba36pxyxel/coolmc.tar?dl=0).
2. Load the docker container: `docker load --input coolmc.tar`
3. Docker workspace initialization (if you want to save the trained policies permanently on your local machine): `bash init_docker_workspace.sh`
4. Run the docker container: With docker workspace initialization: `docker run --user mycoolmc  -v "$(pwd)/prism_files":"/home/mycoolmc/prism_files" -v "$(pwd)/mlruns":"/home/mycoolmc/mlruns" -it coolmc bash`. Without docker workspace initialization: `docker run --user mycoolmc  -v "$(pwd)/prism_files":"/home/mycoolmc/prism_files" -it coolmc bash`


Please make sure that you either run COOL-MC on your machine OR in the docker container. Otherwise, it may lead to folder permission problems.
If there is any problem regarding the docker, it is also possible to download a virtual machine that allows the execution of the docker.
Note that the virtual machine is only for trouble handling and should only be used if the docker container is not executable on your local machine. The virtual machine docker container may not be up to date since the Zenodo repository is immutable and need to be updated via the Getting Started instructions.
Please also make sure that you use the `python` default alias.

We discuss how to create the docker container and how to install the tool natively later.

If you are not familiar with PRISM/Storm, here are some references:

- [PRISM Manual](https://www.prismmodelchecker.org/manual/)
- [Storm Getting Started](https://www.stormchecker.org/getting-started.html)

## Example 1 (Frozen Lake)
Frozen Lake is a commonly used OpenAI Gym benchmark, where 
the agent has to reach the goal (frisbee) on a frozen lake. The movement direction of the agent is uncertain and only depends in $33.33\%$ of the cases on the chosen direction. In $66.66\%$ of the cases, the movement is noisy.

To demonstrate our tool, we will train an RL policy for the OpenAI Gym Frozen Lake environment, verify it, and retrain it in the PRISM environment.
The following command trains the RL policy in the OpenAI Gym FrozenLake-v0 environment.

`python cool_mc.py --task=openai_training --project_name="Frozen Lake Example" --env=FrozenLake-v0 --rl_algorithm=dqn_agent`

Command Line Arguments:
- `task=openai_training` sets the current task to the OpenAI Gym training.
- `project_name=Frozen Lake Example` is the name of YOUR project. Labeling experiments with the same project name allows the comparison between the runs.
- `env=FrozenLake-v0` defines the OpenAI Gym.
- `rl_algorithm=dqn_agent` defines the reinforcement learning algorithm.


After the training, we receive an Experiment ID (e.g. 27f2bbe444754ac0bbbce1326a410419).
We need this ID to identify the experiment.
Now it is possible to verify the trained RL policy via the COOL-MC verifier by passing the experiment ID and the model checking arguments:

`python cool_mc.py --project_name "Frozen Lake Example" --parent_run_id=27f2bbe444754ac0bbbce1326a410419 --task rl_model_checking --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=0" --prop="P=? [F WATER=true]"`

Command Line Arguments:
- `project_name "Frozen Lake Example"` specifies the project name.
- `parent_run_id=XXXX` reference to the trained RL policy (use experiment ID).
- `task=rl_model_checking` sets the current task to model checking the trained RL policy.
- `prism_file_path="frozen_lake3-v1.prism"` specifies the PRISM environment.
- `constant_definitions="control=0.33,start_position=0"` sets the constant definitions for the PRISM environment.
- `prop="P=? [F WATER=true]"` the property query.

It is also possible to plot the property results over a range of PRISM constant definitions. This is useful, when we want to get a overview of the trained RL policy. In the following command, we plot from different frozen lake agent startpositions 0-15 (stepsize 1, 16 excluded) the probability of falling into the water.

`python cool_mc.py --project_name "Frozen Lake Example" --parent_run_id=27f2bbe444754ac0bbbce1326a410419 --task rl_model_checking --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=[0;1;16]" --prop="P=? [F WATER=true]"`

`--constant_definitions="control=0.33,start_position=[0;1;16]"` calculates the property results from `start_position=0` up to `start_position=15` with a stepsize of `1`.


If we are not satisfied with the property result, we can retrain the RL policy via the OpenAI Gym or the PRISM environment. The following command, retrains the RL policy in the PRISM environment. 

`python cool_mc.py --task=safe_training  --parent_run_id=27f2bbe444754ac0bbbce1326a410419  --reward_flag=1 --project_name="Frozen Lake Example" --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=0" --prop="Pmin=? [F WATER=true]"`


- `task=safe_training` specifies the safe training task which allows the RL training in the PRISM environment.
- `prop="Pmin=? [F WATER=true]` tracks the agents probability of falling into the water. `Pmin` also specifies that COOL-MC saves only RL agents which lower probabilities of falling into the water.
- `reward_flag=1` uses rewards instead of penalties.

The following frozen-lake temperature graph shows from each state s the probability for a trained policy of reaching the frisbee at state 15.

![frozen_lake](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/frozen_lake.png)


## Example 2 (Taxi with Fuel)
The taxi agent has to pick up passengers and transport them to their destination without running out of fuel. The environment terminates as soon as the taxi agent does the predefined number of jobs. After the job is done, a new guest spawns randomly at one of the predefined locations.

We train a DQN taxi agent in the PRISM environment:

`python cool_mc.py --task=safe_training --project_name="Taxi with Fuel Example" --rl_algorithm=dqn_agent --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmax=? [F jobs_done=2]"`


- `project_name="Taxi with Fuel Example"` for the new `Taxi with Fuel Example` project.
- `prop="Pmax=? [F jobs_done=2]"` property query for getting the probability to finish 2 jobs with the trained policy. `Pmax` also specifies that COOL-MC save only RL agents which higher probabilities of finishing two jobs.

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="P=? [F \"empty\"]"`

State abstraction allows model checking the trained policy on less precise features without changing the environment. To achieve this, a prepossessing step is applied to the current state in the incremental building process to map the state to a more abstract state for the RL policy. We only have to define a state mapping file and link it to COOL-MC via the command line:

`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="P=? [F \"empty\"]" --abstract_features="../taxi_abstraction.json"`

Permissive Model Checking allows the investigation of the worst-/best-case behaviour of the trained policy for certain state variables.

Minimal Probability of running out of fuel for a fuel level between 4 and 10 (Pmin):
`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F \"empty\"]" --permissive_input="fuel<>=[4;10]"`

Maximal Probability of running out of fuel for a fuel level between 4 and 10 (Pmin):
`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmax=? [F \"empty\"]" --permissive_input="fuel<>=[4;10]"`


## Example 3 (Avoid)
Collision avoidance is an environment which contains one agent and two moving obstacles in a two dimensional grid world. The environment terminates as soon as a collision between the agent and one obstacle happens. The environment contains a slickness parameter, which defines the probability that the agent stays in the same cell.

We train a DQN taxi agent in the PRISM environment:

`python cool_mc.py --task=safe_training --project_name="Avoid Example" --rl_algorithm=dqn_agent --prism_file_path="avoid.prism" --constant_definitions="xMax=4,yMax=4,slickness=0.0" --prop="Pmin=? [F<=100 COLLISION=true]" --reward_flag=1`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=915fd49f5f9342a5b5f124dddfd3f15f --task=rl_model_checking --project_name="Avoid Example" --prism_file_path="avoid.prism" --constant_definitions="xMax=4,yMax=4,slickness=0.0" --prop="P=? [F<=100 COLLISION=true]"`


## Example 4 (Smart Grid)
In this environment, a controller controls the distribution of renewable- and non-renewable energy production. The objective is to minimize the production of non-renewable energy by using renewable and storage technologies.
If there is too much energy in the electricity network, the energy production shuts down which may lead to a blackout (terminal state).


`python cool_mc.py --task=safe_training --project_name="Smart Grid Example" --rl_algorithm=dqn_agent --prism_file_path="smart_grid.prism" --constant_definitions="max_consumption=20,renewable_limit=19,non_renewable_limit=16,grid_upper_bound=25" --prop="Pmin=? [F<=1000 IS_BLACKOUT=true]" --reward_flag=0`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=c0b0a71a334e4873b045858bc5be15ed --task=rl_model_checking --project_name="Smart Grid Example" --prism_file_path="smart_grid.prism" --constant_definitions="max_consumption=20,renewable_limit=19,non_renewable_limit=16,grid_upper_bound=25" --prop="P=? [F<=1000 TOO_MUCH_ENERGY=true]"`

## Example 5 (Warehouse)
In this environment, a controller controls the distribution of renewable- and non-renewable energy production. The objective is to minimize the production of non-renewable energy by using renewable and storage technologies.
If there is too much energy in the electricity network, the energy production shuts down which may lead to a blackout (terminal state).


`python cool_mc.py --task=safe_training --project_name="Warehouse Example" --rl_algorithm=dqn_agent --prism_file_path="storage.prism" --prop="Pmin=? [F<=100 STORAGE_FULL=true]" --reward_flag=0`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=02462a111bf9436d8bcce71a6334d35b --task=rl_model_checking --project_name="Warehouse Example" --prism_file_path="storage.prism" --prop="T=? [F STORAGE_FULL=true]"`

## Example 6 (Stock Market)
This environment is a simplified version of a stock market sim-
ulation. The agent starts with a initial capital and has to increase it through
buying and selling stocks without running into bankruptcy.

We now train a RL policy for the stock market example and try to save the policy with the highest probability of successesfully reaching the maximal amount of capital in 1000 time steps.

`python cool_mc.py --task=safe_training --project_name="Stock Market Example" --rl_algorithm=dqn_agent --prism_file_path="stock_market.prism" --prop="Pmax=? [F<1000 \"success\"]" --reward_flag=1`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=02462a111bf9436d8bcce71a6334d35b --task=rl_model_checking --project_name="Stock Market Example" --prism_file_path="stock_market.prism" --prop="P=? [F<1000 \"bankruptcy\"]"`

## Example 7 (James Bond 007)
This environment is a abstraction of the James Bond game for the Atari 2600. The goal is to collect rewards by shooting helicopters/diamonds and collecting diamonds. James Bond needs to avoid falling into radioactive pixels, which would terminate the environment.
The state space consists of images. Each image consists of three pixel rows with 6 pixels each (3x6), and one extra pixel as auxiliary variable for PRISM.
The actions are jump, tick, and shoot.

`python cool_mc.py --task=safe_training --project_name="James Bond Example" --rl_algorithm=dqn_agent --prism_file_path="james_bond007.prism" --prop="" --reward_flag=1 --max_steps=100 --num_episodes=3000 --seed=128`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=90761357347440c8baf4833b3dcfb330 --task=rl_model_checking --project_name="James Bond Example" --prism_file_path="james_bond007.prism" --prop="P=? [F<=15 done=true]"`

## Example 8 (Crazy Climber)
Crazy climber is a game where the player has to climb up a wall.
This is a PRISM abstraction based on this game.
The game is a grid of pixels.
A pixel with a One indicates the player position.
A pixel with a Zero indicates an empty pixel.
A pixel with a Three indicates a falling object.
A pixel with a Four indicates a collision of the player with a object.
The right side of the wall consists of a window front. The player has to avoid climbing up there since the windows are not stable.
For every level the play climbs, the player gets an reward of 1.
The player can also move left and right to avoid falling obstacles.

We now train a RL policy for the crazy climber example and try to save the policy with the highest average reward.

`python cool_mc.py --task=safe_training --project_name="Crazy Climber Example" --rl_algorithm=dqn_agent --prism_file_path="crazy_climber.prism" --prop="" --reward_flag=1 --max_steps=100 --num_episodes=3000 --seed=128`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=e3d5c0d086fa482bba2ec65f1ba58ad5 --task=rl_model_checking --project_name="Crazy Climber Example" --prism_file_path="crazy_climber.prism" --prop="P=? [F<=15 done=true]"`






## Benchmarks
To replicate the benchmark experiments of our paper, run:

`mlflow run unit_testing --no-conda -e test_experiments`

This command will train and verify the RL policies in the taxi (transporter.prism), collision avoidance (avoid.prism), stock market (stock_market.prism), smart grid (smart_grid.prism), and frozen lake (frozen_lake3-v1.prism) environment.

Use the templates `permissive_policy_plotting.py` and `three_plot_plotting.py` to plot the diagrams from the tool paper.

Furthermore, we also support the PRISM-MDPs of the ![Quantitative Verification Benchmark Set](https://qcomp.org/benchmarks/index.html) as long as they contain reward functions.

## Web-Interface
COOL-MC provides a web interface to analyze the RL training progress and the model checking results.
It also allows the comparison between different trained policies.
First, run `mlflow server -h 0.0.0.0 &` to start the MLFlow server in the background (http://DOCKERIP:5000).

Second, get the IP address of the docker (DOCKERIP) container via `ip add | grep global`.

Third, use your web browser on your local machine to access http://DOCKERIP:5000.

Project Overview with all its experiments:

![web_ui](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/web_ui.png)

Training Progress:

![web_ui](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/training_progress.png)

Compare different policies with each other:

![web_ui](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/comparison.png)


## Model Checking Times and Limitations
All experiments were executed on a NVIDIA GeForce GTX 1060 Mobile GPU, 8 GB RAM, and an Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz x 12.

In the following table, we list the model-building and model checking times for different trained policies.

| Environment    | Constants                                                                        | Property Query                        | Property Result | Model Size | Model Building Time (s) | Model Checking Time (s) |
|----------------|----------------------------------------------------------------------------------|---------------------------------------|--------|------------|-------------------------|-------------------------|
| Taxi with Fuel | MAX_JOBS=2,MAX_FUEL=10                                                           | P=? [F "empty"]                       | 0      | 252        | 3.3                     | 0                       |
| Taxi with Fuel | MAX_JOBS=2,MAX_FUEL=10                                                           | P=? [F jobs_done=2]                   | 1      | 252        | 3                       | 0                       |
| Avoid          | xMax=4,yMax=4,slickness=0                                                        | P=? [F<=100 COLLISION=true]           | 0      | 15133      | 28                      | 0.42                    |
| Avoid          | xMax=4,yMax=4,slickness=0                                                        | P=? [F<=200 COLLISION=true]           | 0      | 15133      | 28                      | 0.5                     |
| Stock Market   |                                                                                  | P=? [F<1000 "bankruptcy"]             | 0      | 130        | 0.2                     | 0                       |
| Smart Grid     | max_consumption=20,renewable_limit=19,non_renewable_limit=16,grid_upper_bound=25 | Pmin=? [F<=1000 TOO_MUCH_ENERGY=true] | 0.99   | 1152       | 3                       | 0.02                    |


COOL-MC needs more time to build the model while using less memory than PRISM/Storm.
The reason for the performance bottleneck is the iterative building of the induced DTMCs.
If we disregard model-building time, we receive roughly the same model checking performance for Storm and COOL-MC.
Our tool generates near-optimal policies w.r.t. to the different properties and builds smaller DTMCs.
Therefore, one major advantage of our tool is that we can model check larger MDPs as with PRISM/Storm.

We can model check the following command with COOL-MC, while it is not possible on our machine with Storm (1077628 states and 118595768 transitions):

`storm --prism "avoid.prism" --constants "xMax=9,yMax=9,slickness=0.1" --prop "Tmin=? [F COLLISION=true]"`

 If, however, we apply the trained policy, the induced DTMC has 100000 states/2574532 transitions, clearly within reach for Storm, while the result may not be optimal:

`python cool_mc.py --task=safe_training --project_name="Avoid Example" --rl_algorithm=dqn_agent --prism_file_path="avoid.prism" --constant_definitions="xMax=9,yMax=9,slickness=0.1" --prop="" --reward_flag=1 --num_episodes=2 -seed=128`

`python cool_mc.py --parent_run_id=a7caa1e60d5e44d583822433c04d902b --task=rl_model_checking --project_name="Avoid Example" --constant_definitions "xMax=9,yMax=9,slickness=0.1" --prism_file_path="avoid.prism" --prop="T=? [F COLLISION=true]"`

T=? [F COLLISION=true] : 291.1856317546715

Model Building Time: 95.92452359199524

Model Checking Time: 2.691767454147339



## PRISM Modelling Tips
We first have to model our RL environment. COOL-MC supports PRISM as a modeling language. It can be difficult to design your own PRISM environments. Here are some tips on how to make sure that your PRISM environment works correctly with COOL-MC:

- Make sure that you only use transition-rewards
- After the agent reaches a terminal state, the storm simulator stops the simulation. Therefore, terminal state transitions will not be executed. So, do not use self-looping terminal states.
- To improve the training performance, try to make all actions at every state available. Otherwise, the agent may choose a not available action and receives a penalty.
- Try to unit test your PRISM environment before RL training. Does it behave as you want?

## RL Agent Training
After we have modeled the environment, we can train RL agents in this environment.
It is also possible to develop your own RL agents:
1. Create an AGENT_NAME.py in the src.rl_agents package
2. Create a class AGENT_NAME and inherit all methods from common.rl_agents.Agent
3. Override all the needed methods (depends on your agent) + the agent save- and load-method.
4. In src.rl_agents.agent_builder extends the build_agent method with an additional elif branch for your agent
5. Add additional command-line arguments in cool_mc.py (if needed)

Here are some tips that may improve the training progress:

- Try to use the disable_state parameter to disable state variables from PRISM, which are only relevant for the PRISM environment architecture.
- Play around with the RL parameters.
- The model checking part while RL training can take time. Therefore, the best way to train and verify your model is first to use reward_max. After the RL model reaches an acceptable reward, change the parameter prop_type to min_prop or max_prop and adjust the evaluation intervals.

## COOL-MC Command Line Arguments
The following list contains all the major COOL-MC command line arguments. It does not contain the arguments which are related to the RL algorithms. For a detailed description, we refer to the common.rl_agents package.

| Argument             | Description                                                                                                                                                                                                                                                                                 | Options                                           | Default Value  |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|----------------|
| task                 | The type of task do you want to perform.                                                                                                                                                                                                                                                    | safe_training, openai_training, rl_model_checking | safe_training  |
| project_name         | The name of your project.                                                                                                                                                                                                                                                                   |                                                   | defaultproject |
| parent_run_id        | Reference to previous experiment for retraining or verification.                                                                                                                                                                                                                            | PROJECT_IDs                                       |                |
| num_episodes         | The number of training episodes.                                                                                                                                                                                                                                                            | INTEGER NUMBER                                    | 1000           |
| eval_interval        | Interval for verification while safe_training.                                                                                                                                                                                                                                              | INTEGER NUMBER                                    | 100            |
| sliding_window_size  | Sliding window size for reward averaging over episodes.                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | 100            |
| rl_algorithm         | The name of the RL algorithm.                                                                                                                                                                                                                                                               | dqn_agent, sarsamax                               | dqn_agent      |
| env                  | openai_training parameter for the environment name.                                                                                                                                                                                                                                         | OPENAI GYM NAMES                                  |                |
| prism_dir            | The directory of the PRISM files.                                                                                                                                                                                                                                                           | PATH                                              | ../prism_files |
| prism_file_path      | The name of the PRISM file.                                                                                                                                                                                                                                                                 | STR                                               |                |
| constant_definitions | Constant definitions seperated by a commata.                                                                                                                                                                                                                                                | For example: xMax=4,yMax=4,slickness=0            |                |
| prop                 | Property Query. **For safe_training:** Pmax tries to save RL policies that have higher probabilities. Pmin tries to save RL policies that have  lower probabilities. **For rl_model_checking:** In the case of induced DTMCs min/max  yield to the same property result (do not remove it). |                                                   |                |
| max_steps            | Maximal steps in the safe gym environment.                                                                                                                                                                                                                                                  |                                                   | 100            |
| disabled_features    | Disable features in the state space.                                                                                                                                                                                                                                                        | FEATURES SEPERATED BY A COMMATA                   |                |
| permissive_input     | It allows the investigation of the worst-/best-case behaviour of the trained policy for certain state variables.                                                                                                                                                                            |                                                   |                |
| abstract_features    | It allows model checking the trained policy on less precise sensors without changing the environment.                                                                                                                                                                                       |                                                   |                |
| wrong_action_penalty | If an action is not available but still chosen by the policy, return a penalty of [DEFINED HERE].                                                                                                                                                                                           |                                                   |                |
| reward_flag          | If true (1), the agent receives rewards instead of penalties.                                                                                                                                                                                                                               |                                                   | 0              |
| range_plotting       | Range Plotting Flag for plotting the range plot on the screen.                                                                                                                                                                                                                              |                                                   |       1        |
| seed                 | Random seed for PyTorch, Numpy, Python.                                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | None (-1)      |


### permissive_input
It allows the investigation of the worst-/best-case behavior of the trained policy for certain state variables. Let's assume we have a formal model with state variables $a=[0,5]$, $b=[0,5]$, $c=[0,2]$. We now want to investigate the permissive policies independent of $c$. Therefore, we generate all the RL policy actions for each state for the different $c$ assignments and incrementally build the MDP. - $c=[0,2]$ generates all the actions for the different c-assignments between $[0,2]$. - $c=[0,1]$ generates all the actions for the different c-assignments between $[0,1]$. - $c<>=[1,2]$ generates all the actions for the different c-assignments [1,2] only if the c value is between $[1,2]$. Otherwise, treat $c$ normally.

### abstract_features
It allows model checking the trained policy on less precise sensors without changing the environment.
To achieve this, a prepossessing step is applied to the current state in the incremental building process to map the state to a more abstract state for the RL policy. We can archive the abstraction by:

- Passing the abstraction mapping file via the command line (e.g. taxi_abstraction.json)
- Passing the abstraction interval via command line (x=[0,2,10], maps the other x-values to the closest abstracted x-value).

This argument is reseted after rerunning the project.



## Manual Installation

### Creating the Docker

You can build the container via `docker build -t coolmc .` It is also possible for UNIX users to run the bash script in the bin-folder.

### Tool Installation
Switch to the repository folder and define environment variable `COOL_MC="$PWD"`

#### (1) Install Dependencies
`sudo apt-get update && sudo apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev python3 python-is-python3 python3-setuptools python3-pip graphviz && sudo apt-get install -y --no-install-recommends maven uuid-dev virtualenv`

#### (2) Install Storm
0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/storm.git`
2. `cd storm`
3. `mkdir build`
4. `cd build`
5. `cmake ..`
6. `make -j 1`

For more information about building Storm, click [here](https://www.stormchecker.org/documentation/obtain-storm/build.html).

For testing the installation, follow the follow steps [here](https://www.stormchecker.org/documentation/obtain-storm/build.html#test-step-optional).

#### (3) Install PyCarl
0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/pycarl.git`
2. `cd pycarl`
3. `python setup.py build_ext --jobs 1 develop`

If permission problems: `sudo chmod 777 /usr/local/lib/python3.8/dist-packages/` and run third command again.


#### (4) Install Stormpy

0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/stormpy.git`
2. `cd stormpy`
3. `python setup.py build_ext --storm-dir "${COOL_MC}/storm/build/" --jobs 1 develop`

For more information about the Stormpy installation, click [here](https://moves-rwth.github.io/stormpy/installation.html#installation-steps).

For testing the installation, follow the steps [here](https://moves-rwth.github.io/stormpy/installation.html#testing-stormpy-installation).

#### (5) Install remaining python packages and create project folder
0. `cd $COOL_MC`
1. `pip install -r requirements.txt`
2. `mkdir projects`

## Reproducing the results of the paper Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking
1. Run the test_experiments to train the RL agents
2. Update the robustness script with the correct parent IDs and run the robustness script
3. Run feature_sensitivity_analysis with the trained RL agent parent IDs to get the property impact results of the property impact attacks.