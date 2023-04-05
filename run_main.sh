# conda activate py38
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# python main.py --model_name ppo
# python main.py --model_name ppo
python main.py --model_name a2c
# python dqn.py
# python fix.py