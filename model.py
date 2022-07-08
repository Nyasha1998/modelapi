# importing necessary libraries and data
from unittest import result
import pandas as pd
import numpy as np
# splitting dataset into training set and testing set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


data = pd.read_csv('data.csv')

# deleting unused columns
data.pop('major')
data.pop('timestamp')
data.pop('gender')
data.pop('level_of_education')

# Converting column categorical values to numeric values 

# defining a function that replaces a text value to a numeric one
def text_to_numeric_values(section, content, value):
    data[section] = data[section].replace({ content: value})
    
text_to_numeric_values('could_not_experience_the_positive_feeling', 'Did not apply to me', 0)
text_to_numeric_values('could_not_experience_the_positive_feeling', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('could_not_experience_the_positive_feeling', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('could_not_experience_the_positive_feeling', 'Applied to me very much or most of the times', 3)

text_to_numeric_values('could_not_work_up_the_initiative_to_do_things', 'Did not apply to me', 0)
text_to_numeric_values('could_not_work_up_the_initiative_to_do_things', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('could_not_work_up_the_initiative_to_do_things', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('could_not_work_up_the_initiative_to_do_things', 'Applied to me very much or most of the times', 3)

text_to_numeric_values('had_nothing_to_look_forward_to', 'Did not apply to me', 0)
text_to_numeric_values('had_nothing_to_look_forward_to', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('had_nothing_to_look_forward_to', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('had_nothing_to_look_forward_to', 'Applied to me very much or most of the times', 3)

text_to_numeric_values('felt_down_hearted_and_blue', 'Did not apply to me', 0)
text_to_numeric_values('felt_down_hearted_and_blue', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('felt_down_hearted_and_blue', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('felt_down_hearted_and_blue', 'Applied to me very much or most of the times', 3)

text_to_numeric_values('were_unable_to_become_enthusiastic', 'Did not apply to me', 0)
text_to_numeric_values('were_unable_to_become_enthusiastic', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('were_unable_to_become_enthusiastic', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('were_unable_to_become_enthusiastic', 'Applied to me very much or most of the times', 3)

text_to_numeric_values('felt_you_werent_worth_much_as_a_person', 'Did not apply to me', 0)
text_to_numeric_values('felt_you_werent_worth_much_as_a_person', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('felt_you_werent_worth_much_as_a_person', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('felt_you_werent_worth_much_as_a_person', 'Applied to me very much or most of the times', 3)

text_to_numeric_values('felt_life_was_meaningless', 'Did not apply to me', 0)
text_to_numeric_values('felt_life_was_meaningless', 'Applied to me to some degree, or some of the time', 1)
text_to_numeric_values('felt_life_was_meaningless', 'Applied to me to a considerable degree or a good part of time', 2)
text_to_numeric_values('felt_life_was_meaningless', 'Applied to me very much or most of the times', 3)


# defining independent variables x, and dependent variable y
x = data.drop(['score'], axis=1).values
y = data[["score"]].values

# splitting dataset into training set and testing set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# Train the linear model 
ml=LinearRegression()
model = ml.fit(x_train, y_train)

# predict the test results
y_predict=ml.predict(x_test)

# save the model to disk
filename = 'model.sav'
joblib.dump(model, filename)
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)