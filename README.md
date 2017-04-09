# Bank_Fears_Loanliness

Language used - Python 3

Libraries used 

Pandas
Numpy
Scikit-learn

Developed in jupyter

My approach is to do a basic analysis of the dataset provided. After the base exploration I dropped the columns with empty values or the columns which had a large percentage of one type of value, as it wouldn't add much information to the model generated. I used scikit-learn's Random Forest Regressor to predict the classes for the test dataset. 
For initial runs the dataset was split in to the training and test datasets. The accuracy on the test dataset was measured by the website and the results for this dataset weren't made public. This model achieved an accuracy of 97% on the test dataset according to the website. The test dataset cannot be tested as the correct results are NOT available. 


This repo contains both the Jupyter notebook which explains the steps and the rationale behind the decisions made during the model generation and a .py file for anyone who wants to see just the code. 

If the data is not available at the link mentioned in the notebook, you can get the dataset from this link.
https://drive.google.com/file/d/0B91JEPO_jfR4d1ZSZGZiUXBRU1U/view?usp=sharing
