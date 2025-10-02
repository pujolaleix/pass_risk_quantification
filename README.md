# Pass Risk Quantification Metric
This works processes and analyses StatsBomb data to create a metric that quantifies pass risk. Results are displayed using Italy national team matches in Euro2020 competition, where they achieved their second UEFA EURO title.
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
