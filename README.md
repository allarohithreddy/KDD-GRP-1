Deliverable 1

Team Members:

Julie Berryhill
Manoj Aitha
Gabriel Van Dreel
Tammy Ziegler
Dasha Rizvanova

Project Introduction:
Group 4 decided that they would like to research the possibility of whether an individual could change their lifestyle factors in time to reduce or eliminate the onset of Alzheimer's. For example, if an individual knew that a healthier diet could reduce their chance of a positive diagnosis, then they may put eating healthier as an earlier priority in their lives. This dataset includes many lifestyle factors (BMI, smoking status, alcohol consumption, physical activity, diet quality, and sleep quality) that will help in building predictive and prescriptive objectives.
We will be using supervised learning methods as the diagnosis of Alzheimer's is a known outcome in this dataset. We will be looking into classification, clustering, and feature selection to identify and select the most relevant features to improve the predictive models. By using these approaches, group 4 hopes that this research will uncover any insights of lifestyle risk factors for developing Alzheimer’s and which lifestyle factor modifications could mitigate those risks.
Research Question:

Are there any lifestyle factors that could be changed to reduce or eliminate the onset of Alzheimers?
https://www.cdc.gov/aging/publications/features/lower-your-dementia-risk/index.html
https://www.medicalnewstoday.com/articles/4-lifestyle-changes-may-improve-cognitive-function-slow-alzheimers
https://edition.cnn.com/2024/06/07/health/alzheimers-dementia-ornish-lifestyle-wellness/index.html
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset?resource=download



Gabriel Van Dreel: What are the most influential predictors for a diagnosis of Alzheimer's disease?

Julie Berryhill: Are people more at risk based on gender and/or ethnicity? Are there reasons to believe that physically fit individuals with a healthier diet are less at risk?

Dasha Rizvanova: 

How do factors such as alcohol consumption, age, ethnic background, and medical history influence the risk of being diagnosed with Alzheimer's Disease? 

Most importantly, can we develop a model to identify individuals at risk based on these factors?

Tammy Questions
What characteristics are most common in people who become diagnosed with Alzheimer's disease?


Which of the characteristics can be identified prior to the onset of the disease?
https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8927715/
Dasha Questions
How do factors such as alcohol consumption, age, ethnic background, and medical history influence the risk of being diagnosed with Alzheimer's Disease? Most importantly, can we develop a model to identify individuals at risk based on these factors?

 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10253673/ https://www.washingtonpost.com/wellness/2024/03/27/dementia-aging-risk-brain-diabetes-pollution-alcohol/



Relevant Domain Information:

Data Source and Description:

The Alzheimer’s Disease Dataset was selected:
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset?select=alzheimers_disease_data.csv

The dataset consists of 2150 entries detailing various physiological characteristics of patients who were considered for an Alzheimer’s disease diagnosis and whether they were formally diagnosed with the illness. Some of the data such as the gender and ethnicity columns is categorical while other data is numeric and represents either an index for a rating of some condition such as sleep quality or a direct measurement of some physical characteristic such as cholesterol. All of the numeric data points could be considered time series data with respect to age. Some columns in the dataset such as the name of the doctor in charge of a given patient, however, have been sanitized and provide no meaningful information.

Data Understanding and EDA:

We learned that Lifestyle and Cognitive features seem to have more of a correlation for Alzheimer's prediction.


Data Preparation:

The categorical columns of the Alzheimer’s Disease Dataset could be one-hot encoded during data preprocessing. Any of the numeric columns could be standardized such that they reflect a normal distribution if they are found to be normally distributed while others could be linearly scaled to reduce the effect of bias in any machine learning models for the dataset. The sanitized columns in the dataset could also be dropped since they provide no useful information.


Citation
If you use this dataset in your work, please cite it as follows:
@misc{rabie_el_kharoua_2024,
title={Alzheimer's Disease Dataset},
url={https://www.kaggle.com/dsv/8668279},
DOI={10.34740/KAGGLE/DSV/8668279},
publisher={Kaggle},
author={Rabie El Kharoua},
year={2024}
}


