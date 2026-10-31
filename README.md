# Bank Churn Prediction

A machine learning project that identifies bank customers who are likely to leave, so the bank can contact them before they go.

**Tools:** Python, pandas, scikit-learn, XGBoost

**Author:** Solace Asante

## Results

The final model is XGBoost with a decision threshold of 0.15. It was evaluated once on 2,000 held-out customers. It identifies most customers at risk of leaving, giving the bank more opportunities to retain them.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.867 |
| Recall | 83.5% |
| Precision | 43.1% |
| Churners identified | 340 of 407 |
| Customers flagged | 789 of 2,000 |

## Dataset

The project uses the Kaggle [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling). It contains 10,000 customer records, 10 input features and a target variable, `Exited`.

The dataset is not included in this repository. Place `Churn_Modelling.csv` in `data/raw/`.

**License:** Not specified on the Kaggle page.

## Methodology

The project began with data cleaning and exploratory analysis. Identifier columns were removed and categorical variables were encoded.

Exploratory analysis showed that age and the number of products have non-linear relationships with churn. Grouping customers by age and treating the number of products as a categorical feature raised logistic regression recall from 21% to 45%.

Four models were compared: logistic regression, decision tree, random forest and XGBoost. They were tuned with grid search using stratified 5-fold cross-validation on the training set. The test set was used once, at the end. Performance was measured with ROC-AUC, precision and recall, rather than accuracy alone, because most customers did not churn.

XGBoost was selected as the final model because use it  had the highest cross-validation AUC of 0.866 which was slightly  ahead of the random forest's 0.860

In cross-validation, the default threshold of 0.50 gave 48.0% recall and 75.8% precision. Lowering it to 0.15 raised recall to 82.1% and reduced precision to 44.3%. The lower threshold was chosen because identifying more potential churners was the main business priority, even if it meant flagging more customers who would have stayed.

## Installation and Usage

Create a virtual environment and install the required packages

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the model 

```bash
python src/train.py
```

Generate predictions for new customers

```bash
python src/predict.py path/to/customers.csv
```

## Project Structure

```text
bank-churn-prediction/
├── data/
│   ├── raw/             
│   └── processed/       
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── clean.py
│   ├── train.py
│   └── predict.py
├── models/              
├── requirements.txt
└── README.md
```

## Limitations

The dataset contains no dates or customer history, so time-based validation was not possible. 
The test result comes from a model trained on 80% of the data, while the saved model is refit on all rows. 

Gender is included as a feature in this learning project and is not intended for use in real banking decisions. Retention costs and customer lifetime values were  unavailable, so the 0.15 threshold was chosen to catch more churners rather than based on actual business costs.

## Disclaimer

This is a learning project and is not intended for production use.
