# Project 04: Disease And Treatment

## Project Brief
The goal is to gather and analyze data on the mosquito population to inform public health strategies. Effective pesticide deployment is crucial for controlling the virus spread, but it comes with high costs. This project aims to develop a data-driven approach to optimize pesticide use across the city.

## Objective
The primary objective is to develop a predictive model that can estimate the likelihood of mosquitos testing positive for the West Nile virus. The model will leverage data on weather, location, mosquito testing results, and pesticide application. By accurately predicting virus hotspots, the model will help the City of Chicago and the Chicago Department of Public Health (CDPH) deploy pesticides more efficiently, enhancing public health outcomes while managing expenses.

## Findings and Evaluation
### Exploratory Data Analysis (EDA) Findings
- The EDA revealed significant insights into the relationship between weather conditions and the presence of the West Nile Virus (WNV) in mosquitoes. Key findings include:
  - Temperature and precipitation are critical factors influencing mosquito populations and WNV presence.
  - Certain mosquito species are more likely to carry WNV, with notable differences observed across species.
  - The timing of pesticide sprays and their frequency significantly impact the reduction of WNV-positive mosquitoes.
  - Geospatial analysis indicated specific hotspots within Chicago where WNV presence was consistently higher, guiding targeted interventions.

### Model Evaluation and Findings
- Multiple models were evaluated to predict the presence of WNV, including:
  - **XGB Classifier**: Achieved the highest overall performance with a cross-validation mean score of 0.85, a validation accuracy of 0.946, and a ROC AUC score of 0.87. This model effectively balances sensitivity and specificity.
  - **CatBoost Classifier**: Demonstrated strong performance with a mean cross-validation score of 0.84 and a high ROC AUC score, making it a viable alternative.
  - **LightGBM Classifier**: Provided competitive accuracy and faster training times, with a mean cross-validation score of 0.83.
  - **Random Forest Classifier**: Showed reasonable performance but was prone to overfitting, with a lower mean cross-validation score compared to training accuracy.
  - **AdaBoost Classifier**: Improved accuracy by combining multiple weak classifiers, but did not match the XGB Classifier in terms of performance metrics.
- The optimized XGBClassifier is recommended for deployment due to its superior performance metrics. Future work could involve further hyperparameter tuning, feature engineering, and testing the model on real-world data to ensure robustness and generalizability.

## Presentation Slide
[Project Presentation](https://docs.google.com/presentation/d/1ayRNaXmC1KZ2o21sywcMrnGyUqF42Ifm_ZBHtPsVIXM/edit?usp=sharing)

## Dataset
The dataset, along with description, can be found here: [Kaggle - Predict West Nile Virus](https://www.kaggle.com/c/predict-west-nile-virus/).

## Content
The project is structured as follows:

1. **Data Preparation and Cleaning**
   - Import libraries:
     - Pandas and Numpy
     - Libraries for Visualization
     - SK learn Libraries
     - Save models
   - Import data and data dictionaries:
     - train.csv
     - weather.csv
     - spray.csv
     - test.csv
   - Data Cleaning:
     - Check missing values for all datasets
     - Clean up weather dataset
     - Drop unnecessary columns
     - One hot encoding for Codesum
     - Convert measurement types
     - Clean up spray_df
     - Convert Date to python datetime series
     - Add spray count to the dataframe
   - Export cleaned data for EDA and Feature Engineering
   - Create checkpoint and save cleaned data

2. **Exploratory Data Analysis (EDA)**
   - Merge Train and weather dataframes
   - Perform exploratory analysis to understand the data

3. **Feature Engineering and Model Development**
   - Feature Engineering:
     - Split the weather station into two
     - Export CSV for Modelling
   - Model Development
   - Model Prediction on test set

4. **Conclusion**
   - Among the models evaluated, the optimized XGBClassifier achieved the highest overall performance, making it the most suitable for predicting the presence of the West Nile virus in this dataset.
   - The CatBoostClassifier and LightGBMClassifier also demonstrated strong performance and could be considered as viable alternatives.
   - RandomForestClassifier and AdaBoostClassifier provided reasonable performance but did not match the optimized XGBClassifier in terms of accuracy and ROC AUC score.
   - Overall, the optimized XGBClassifier is recommended for deployment due to its superior performance metrics. Future work could involve further hyperparameter tuning, feature engineering, and testing the model on real-world data to ensure robustness and generalizability.

## Running the Code
To run the code in this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries listed in the `requirements.txt` file.
3. Open the `MAIN.ipynb` notebook in Jupyter Notebook or Jupyter Lab.
4. Run the notebook cells in order to execute the data preparation, EDA, feature engineering, and model development steps.

## Running the Streamlit App
To run the Streamlit app, follow these steps:

1. Ensure you have Streamlit installed. If not, install it using the following command:
   ```bash
   pip install streamlit
   ```
2. Navigate to the directory containing the `streamlit_app.py` file.
3. Run the Streamlit app using the following command:
   ```bash
   streamlit run streamlit_app.py
   ```
4. The app will open in your default web browser. You can input longitude, latitude, and month to get a predicted probability score of the presence of the West Nile virus.

## Repository Structure
- `MAIN.ipynb`: The main Jupyter Notebook containing the project code.
- `streamlit_app.py`: The Streamlit app script.
- `data/`: Directory containing the datasets used in the project.
- `models/`: Directory to save the trained models.
- `output/`: Directory to save the output files and results.

## Online Research Material
### Data Source
- [CDC West Nile Virus Historic Data](https://www.cdc.gov/west-nile-virus/data-maps/historic-data.html)
- [CDC West Nile Virus Data and Maps](https://www.cdc.gov/west-nile-virus/data-maps/index.html)
- [City of Chicago - Preventing West Nile Virus](https://www.chicago.gov/city/en/depts/cdph/supp_info/infectious/preventing_west_nilevirus.html)
- [CDC - West Nile Virus Prevention](https://www.cdc.gov/west-nile-virus/prevention/index.html)

### Spray Cost
- [Mosquito Treatment Price - LawnStarter](https://www.lawnstarter.com/blog/cost/mosquito-treatment-price/)
- [Cost of Mosquito Treatment - Angi](https://www.angi.com/articles/cost-mosquito-treatment.htm)

### West Nile Virus Outbreak Research
- [West Nile Virus - AAFP](https://www.aafp.org/pubs/afp/issues/2003/0815/p653.html)
- [Research on West Nile Virus - NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3111838/)
- [West Nile Virus in Indiana - 2002](https://www.in.gov/health/reports/disease/2002/west_nile.htm)

### Mosquitoes Variant
- [Culex Pipiens and Culex Restuans](https://azelisaes-us.com/grow_your_know/culex-pipiens-culex-restuans/)
```
