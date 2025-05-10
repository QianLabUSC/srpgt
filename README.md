# Safe Reactive Navigation for Granular Terrain Exploration

## Setup Virtual Environment Using Conda

```
git clone git@github.com:matthewyjiang/reactive-navigation.git -b revision2
conda create -n reactivenav python=3.9
conda activate reactivenav
cd reactive-navigation
pip install -r requirements.txt

cd ..
git clone git@github.com:matthewyjiang/SafeOpt.git
cd SafeOpt
python3 setup.py install
```

## Run code using 

```python3 main.py```

- Space bar to start/stop simulation

- Click to set goal location
