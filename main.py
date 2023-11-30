from data_loader import load_data
from data_preprocessing import normalization_options
from feature_selection import feature_selection_options
from cross_validation import cross_validation_options
from classification_models import model_selection_options
from report_generator2 import generate_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Loading the csv dataset
data = load_data('dataset.csv')  # data_loader
print("Handling categorical values")
# To know the columns which has categorical values 
categorical_columns = data.select_dtypes(include=['object']).columns 
# Converting categorical values to numerical values
label_encoders = {} 
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("Handling missing values")
# Replacing null/missing values with the mean values of that column
for column in data.columns: 
    if data[column].isnull().any():
        mean_value = data[column].mean()
        data[column].fillna(mean_value, inplace=True)

# Selecting the last column as class or label
target = data[data.columns[-1]] 
# Selecting data other than last column
data = data.drop(data.columns[-1], axis=1) 



# Taking inputs for different methods
normalization_method = input("Select normalization/standardization method (Standard/MinMax/Robust): ")
feature_selection_method = input("Select feature selection method (SelectKBest/RFE/VarianceThreshold/PCA/SFS): ")
cross_validation_method = input("Select cross-validation method (kfold/stratified_kfold/leave_one_out): ")
model_type = input("Select classification model (RandomForest/SVM/LogisticRegression/GradientBoosting/NaiveBayes): ")

normalized_data = normalization_options(data, method=normalization_method) # data_preprocessing
selected_data = feature_selection_options(normalized_data, target, method=feature_selection_method) # feature_selection

# Splitting the dataset into training dataset and blind dataset
x_train, x_blind, y_train, y_blind = train_test_split(selected_data, target, test_size=0.1, random_state=42) 

model = model_selection_options(model_type=model_type) # classification_models
scores_accuracy, scores_precision, scores_recall, scores_f1, best_model = cross_validation_options(model, x_train, y_train, cv_method=cross_validation_method) # cross_validation

print(scores_accuracy)
print(scores_precision)
print(scores_recall)
print(scores_f1)

print("Testing our model on Blind Dataset....\n")

y_pred = best_model.predict(x_blind)
blind_accuracy = accuracy_score(y_blind, y_pred)
print(blind_accuracy)
blind_precision = precision_score(y_blind, y_pred, average = None)
print(blind_precision.mean())
blind_recall = recall_score(y_blind, y_pred, average = None)
print(blind_recall.mean())
blind_f1 = f1_score(y_blind, y_pred, average = None)
print(blind_f1.mean())

kin = [1, 2, 3, 4, 5] 


blind_metrics = {
    'accuracy': round(blind_accuracy, 2),
    'precision': round(blind_precision.mean(), 2),
    'recall': round(blind_recall.mean(), 2),
    'f1_score': round(blind_f1.mean(), 2)
}
generate_report(kin, scores_accuracy, scores_precision, scores_recall, scores_f1, blind_metrics , best_model, feature_selection_method, normalization_method, cross_validation_method)  