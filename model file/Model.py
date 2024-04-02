from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

def train_model(data_file_path):
    # Load the dataset
    data = pd.read_csv("C:\Users\rashe\.vscode\.vscode\project\student_performance_project\student_exam_data.csv")  # Adjust based on your dataset format (e.g., CSV, Excel, etc.)

    # Assuming 'target' is the column you want to predict, and 'feature1', 'feature2', etc., are your input features
    features = data[['Study Hours', 'Previous Exam Score']]  # Adjust features based on your dataset
    target = data['Pass/Fail']  # Adjust target based on your dataset

    # Split the dataset into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(features_train, target_train)

    # Save the trained model
    joblib.dump(model, 'trained_model.joblib')

    # Print model training accuracy (you might want to use a proper evaluation metric based on your problem)
    train_accuracy = model.score(features_train, target_train)
    print(f'Training Accuracy: {train_accuracy}')

    # Print model testing accuracy (you might want to use a proper evaluation metric based on your problem)
    test_accuracy = model.score(features_test, target_test)
    print(f'Testing Accuracy: {test_accuracy}')

# Provide the path to your dataset file
data_file_path = "C:\Users\rashe\.vscode\.vscode\project\student_performance_project\student_exam_data.csv"

# Train the model using the specified dataset
train_model(data_file_path)






# model.py

#/
#from sklearn.linear_model import LinearRegression
#import joblib
#import pandas as pd

#def train_model(data):
    # Dummy training function using a simple linear regression model
    #features = data[['Study Hours', 'Previous Exam Score','Pass/Fail']]  # Adjust features based on your dataset
    #target = data["C:\Users\rashe\student_exam_data.csv"]  # Adjust target based on your dataset

   # model = LinearRegression()
  #  model.fit(features, target)

    # Save the trained model
 #   joblib.dump(model, 'trained_model.joblib')

# Dummy data for training (replace this with your actual dataset)
#data_for_training = pd.DataFrame({
   # 'Study Hours': [1],
  #  'Previous Exam Score': [2],
 #   'target': [3]
#})

#train_model(data_for_training)
# model.py

