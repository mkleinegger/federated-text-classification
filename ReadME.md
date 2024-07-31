# Federated Learning for Text classification
This repository contains all the necessary code, which was used to perform the analysis

## Setup
Run the following commands to get an ready to go environment

    chmod +x ./setup.sh
    ./setup.sh

Aftward, select within `notebook.ipynb` the python environment `federated-learning`.

## Expirments
We ran the expirements on a M1 Macbook Pro with 16GB of RAM. We used 100 epochs to train the state-of-the-art baseline, and the model we want to federate. 
All the necessary code can be found within `notebook.ipynb`. The results are saved under `./results` and the plots are generated with `analysis.ipynb`.

### Config
You can set all parameters of the model, within the first cell of the jupyter notebook. The default values were used to conduct the experiments