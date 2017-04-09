

# importing basic libraries we need to do our analysis
import pandas as pd
import numpy as np
# sklearn library for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Let's start by reading in the data file and understanding what our data looks like.
print("Reading files")
train_df = pd.read_csv("train_indessa.csv")
test_df = pd.read_csv("test_indessa.csv")

# Dropping columns
def droppingCols(df):
    # stripping " months" from each value in column 'term'
    df['term'] = df['term'].map(lambda x: x.rstrip(' months'))
    #converting to int
    df['term'] = pd.to_numeric(df['term'], errors='ignore')
    # multiplying by 4 to convert to weeks
    df['term'] *= 4
    # removed "th week" from each in value in 'last_week_pay', already in weeks so no need to convert
    df['last_week_pay'] = df['last_week_pay'].map(lambda x: x.rstrip('th week'))

    # Removing < and +years from all the emp_length values
    df['emp_length'] = df['emp_length'].map(lambda x: x.rstrip('+ years, year'))
    # Replacing n/ with np.null for further processing And changing < 1 = 0
    df['emp_length'].replace(to_replace = "< 1", value = 0, inplace = True)
    df['emp_length'].replace(to_replace = "n/", value = np.nan, inplace = True)

    # from our exploration we have selected the following columns to remove from our dataset
    columnToRemove = ['batch_enrolled', 'grade', 'sub_grade', 'emp_title', 'verification_status', 'desc', 
                      'purpose', 'title', 'zip_code', 'addr_state', 'mths_since_last_delinq', 'mths_since_last_record', 
                      'total_rec_late_fee', 'mths_since_last_major_derog', 'verification_status_joint']
    # only do once in each run as it will give an error if done without loading the dataset again (columns already removed)
    df.drop(columnToRemove, axis = 1, inplace = True)
    return df

# Creating a new column in our df
# money_bank_paid = funded_amnt - funded_amnt_inv (Substracting in this way to not have negative values)
def addCols(df):
    df['money_bank_paid'] = df['funded_amnt'] - df['funded_amnt_inv']

    # Converting string based abstract classes to category type
    df['home_ownership'] = df['home_ownership'].astype('category')
    df['pymnt_plan'] = df['pymnt_plan'].astype('category')
    df['initial_list_status'] = df['initial_list_status'].astype('category')
    df['application_type'] = df['application_type'].astype('category')

    # all columsn with dtype category.
    categoryColumns = df.select_dtypes(['category']).columns
    df[categoryColumns] = df[categoryColumns].apply(lambda x: x.cat.codes)

    # way to convert a single column
    #df['home_ownership'] = df['home_ownership'].cat.rename_categories(range(len(df['home_ownership'].cat.categories)))
    return df


# replace empty woth 0s
def replaceEmpty(df):
    # replacing the empty spaces with nan values and nan with 0
    df = df.replace(r'\s+', np.nan, regex=True)
    df = df.replace(np.nan, 0, regex=True)
    df = df.replace('NA', 0, regex=True)
    return df


# basic exploration of a column of our dataset. Looking at three different things
def studyColumn(df, columnName):
    # different ways of looking at unique values in a column
    #df["member_id"].unique()
    print("Unique values in column are: ", df[columnName].unique())
    # counting the total number of null or nan values
    print("Total number of null values is: ", df[columnName].isnull().sum())
    # counting th  total number of each type of value in a column
    print("The frequency of each unique value is: ", df[columnName].value_counts(normalize = True))

#studyColumn(train_df, "emp_length")

print("Cleaning data")
# calling the changes on both the dataframes
# first for the trainig dataset
train_df = droppingCols(train_df)
train_df = addCols(train_df)
train_df = replaceEmpty(train_df)

test_df = droppingCols(test_df)
test_df = addCols(test_df)
test_df = replaceEmpty(test_df)

Y_train = train_df['loan_status']
X_train = train_df.drop('loan_status', axis = 1)

print("Training model")
# We are using the random forest for this problem
RFmodel = RandomForestRegressor(n_estimators=10)
# fitting the model on the training dataset
RFmodel.fit(X_train, Y_train)

print("Predicting...")
# Now that we've our model we can test it on our test dataset for predicting the results
Y_pred = RFmodel.predict(test_df)

print("Writing File")
temp = pd.DataFrame(test_df['member_id'])
temp['loan_status'] = Y_pred
temp.reset_index(level = None, drop = True, inplace = True)
temp['loan_status'].replace(0, 0.01,inplace=True)
temp['loan_status'].replace(1, 0.99,inplace=True)
temp.to_csv("submission.csv", sep = ",", index = False)
print("submission.csv created with the results")
