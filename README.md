
# Capstone Project - Azure Machine Learning Engineer

In this project, we are asked to create a model and it's pipeline using Azure Machine Learning Studio and then deploy the best model and consume it. For this, we will be using two approaches in this project to create a model:
AutoML
HyperDrive
And the best model from the above methods will be deployed. 
We will be using a LogisticRegression classifier for training the model and accuracy as a metric for checking the best model.

## Dataset
Pima Indians Diabetes Dataset
https://www.kaggle.com/uciml/pima-indians-diabetes-database

### Overview

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The datasets consist of several medical predictor variables and one target variable, Outcome. Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

The details of Nine columns is as follows:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)

### Task
In this project our task is to predict whether a user is diabetic or not based on features.


Following steps:
1. This project requires creating compute instance to run Jupyter Notebook & compute cluster to run the experiments.
 
2. Dataset needs to be manually uploaded.
 
3. Experiments were run using Auto-ML & HyperDrive.
 
4. The best model was deployed and consumed that gave good metrics.


### Access
I uploaded the dataset in the Azure Machine Learning Studio in Datasets tab through the 'upload from local file' option. The dataset is given the name ‘Diabetes’

![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/Dataset%20d.JPG)
## Automated ML
I have given following  settings for AutoML run

automl_settings = {"experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'accuracy',
    "n_cross_validations": 3,
    "iterations": 24
}


I have given following configuration for AutoML run

automl_config = AutoMLConfig(compute_target=compute_target,
                             task = 'classification',
                             training_data=dataset,
                             label_column_name='Column9',
                             enable_early_stopping= True,
                             featurization = 'auto',
                             debug_log = 'automl_errors.log',
                            )

### Results
Following models were trained 
![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/models%20trained%201.JPG)
![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/models%20trained2.JPG)

The  Model with highest accuracy was Voting Ensemble Model with accuracy of 0.66449 (66.449%).

![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/automl%20model.JPG)

![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/accuracy%20of%20automl.png)

Run Deatils

![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/run%20details.JPG)
![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/run%20details1.JPG)

 best model trained 
 ![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/best%20model%20automl.JPG)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
For HyperDrive I have chosen the LogisticRegression classifier as a model because our is a classification problem. Our target column of the dataset is to predict whether a person is diabetic or not (i.e. 1 or 0). the model is trained using the script written in 'train.py' file.

I have used following parameters

param_sampling = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,0.5,1.0,10.0,50.0,100,1000),
        '--max_iter': choice(10,25)
    }
)

### Results
Following =are the Run details for HyperDrive with Parameter details

![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/hyperdrive%20run%20f.png)

Best performing model has a 63.958% accuracy rate with --C = 1000 and --max_iter = 25.

Run details 
![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/run%20details%20hyper%20f.png)
![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/hyper%20graph.png)


![alt text](https://github.com/NikitaMahajan19/Capstone---Azure-Machine-Learning-Engineer/blob/master/images/hyper%20graph%202.png)


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
