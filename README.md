# Continual learning for Human Activity Recognition

Human Activity Recognition (HAR) has been developed for a long time. Many highly accurate models were proposed. However, one major obstacle towards the real-life smart home is the ability to adapt to change. 

This project aims to develop a continual learning model that is able to learn a new task but also retain previously learned tasks. The key technique used here is **generative replay**. Two agents are working together: one generative model is to mimic all trained lesson while one fully-connected neural network is to gradually lean to capture structured knowledge.

See technical detail [here](Reports/final-report.pdf)


It is an part of [CS5099 Dissertation](https://info.cs.st-andrews.ac.uk/student-handbook/modules/CS5099.html) at University of St. Andrews.


## Set up

1. Prepare python environment `python3 -m venv env`
1. Enter to the env `source env/bin/activate`
2. Install library `pip install -r requirements.txt`

## Run experiments

Run `python run_main.py --results-dir="YOUR RESULT DIR" --data-dir="DATASET"`

`DATASET` must be one of these options \["casas", "housea", "pamap", "dsads"\]

Other available arguments can be seen by running `python run_main.py --help`

You could run `run_*.py` for a different experiment (The same arguments can be applyied in any experiments). You need to run `visdom` if you want to run  `run_plot.py`.

## Visdom

1. Run `python3 -m visdom.server` in one terminal
2. Open browser and then enter url given by the visdom server
1. Open anohter terminal then run your model with `--visdom` (only work with `run_plot.py`)
2. See the interactive report

### Todos
* implement XdG
* consider diversity as well as accuracy
    * self-verifying => select samples with probability rather than yes/no result
    * implement Minibatch or Feature mathcing to avoid mode collapse
    * try Unrolled loss https://arxiv.org/abs/1611.02163
* automated parameter selection
    * fANOVA
* try other GANs
* try hierarchy model
* experiments with task evolution
* experiments with noise; robustness to noise


## Author
[Pakawat Nakwijit](http://curve.in.th); An ordinary programmer who would like to share and challange himself. It is a part of my 2018 tasks to open source every projects in my old treasure chest with some good documentation. 

## License
This project is licensed under the terms of the MIT license.




