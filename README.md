# Pass Risk Quantification Metric
This works processes and analyses StatsBomb data to create a metric that quantifies pass risk. Information from ``events.json``, ``lineups.json`` and ``three-sixty.json`` StatsBomb files is extracted, processed and transformed to build metrics to create **Pass Risk Measure**. Metrics considered are *length*, *passer_pressure*, *passer_pressure_dist*, *receiver_pressure*, *receiver_pressure_dist*, and *bypassed_opponents* which are normalized and combined with user-configurated weighting to compute risk metric. As a starting point, I have used the following configuration:

$ \mathrm{risk}_i = length*0.3 + passer_pressure*0.1 + passer_pressure_dist*0.1 + receiver_pressure*0.1 + receiver_pressure_dist*0.1 + bypassed_opponents*0.3 $

Results are displayed using Italy national team matches in Euro2020 competition, where they achieved their second UEFA EURO title. Below visual represent the pass location and risk quantification of each pass in the final Euro2020 match between Italy (in blue) and England (in white). Completed passes are represented with green arrows, whereas missed/intercepted are displayed in red.

<p align="center">
  <img src="euro2020_final_pass_animation.gif" width="500" alt="Pass risk animation">
</p>


## Scripts
- **download_italy_euro2020_data.py**  
  Utility script to download all available information from the [StatsBomb open-data repository](https://github.com/statsbomb/open-data) related to Italy’s matches in Euro 2020 competition.

  Files are saved under `data/italy_euro2020` directory, maintaining original StatsBomb repository structure.

- **auxiliar_functions.py**
  Contains functions necessary to preprocess and transform data, perform feature engineering, and visualize results. They are separated from the main `pass_risk.ipynb` file for a better readability.

- **pass_risk.ipynb**
  This is the main file that contains end-to-end pipeline to create and visualize pass risk quantification.
  


## Data Sources
This project uses data provided by [StatsBomb](https://statsbomb.com).

Please refer to their [Open Data License](https://github.com/statsbomb/open-data/blob/master/LICENSE.md) for terms. 
