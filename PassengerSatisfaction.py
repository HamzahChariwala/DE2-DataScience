import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.cm as cm

train_data = pd.read_csv("Desktop/DataScience/train.csv")
test_data = pd.read_csv("Desktop/DataScience/test.csv")

train_df = train_data.copy()
test_df = test_data.copy()
train_df, validation_df = train_test_split(train_df, test_size=0.25, random_state=2002)

TARGET = "satisfaction"

# print(f"The train dataset has {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
# print(f"The test dataset has {test_df.shape[0]} rows and {test_df.shape[1]} columns.")
# print(f"The validation dataset has {validation_df.shape[0]} rows and {validation_df.shape[1]} columns.")

# print(train_df.head())

# train_df.info()

# print(train_df.isna().sum())
# print(test_df.isna().sum())
# print(validation_df.isna().sum())

# print(train_df.describe().transpose())

train_df.drop(columns=["Unnamed: 0", "id"], axis=1, inplace=True)
test_df.drop(columns=["Unnamed: 0", "id"], axis=1, inplace=True)
validation_df.drop(columns=["Unnamed: 0", "id"], axis=1, inplace=True)

train_array = train_df.transpose().values
column_headings = list(train_df.columns)

def categorise_columns(data, threshold):
    discrete = []
    continuous = []
    for i in range(len(data)):
        if len(set(data[i]))/len(data[i]) < threshold:
            discrete.append(i)
        else: continuous.append(i)
    return discrete, continuous

discrete_columns, continuous_columns = categorise_columns(train_array, 0.0001)


def plot_discrete_data(df: pd.DataFrame, headings, indices):

    discrete_columns = [headings[i] for i in indices]
    colour_map = sns.color_palette("rocket")

    for i, col in enumerate(discrete_columns):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        sns.countplot(x=col, data=df, ax=axes[0], palette=colour_map)
        axes[0].set_title(f'Bar Chart of {col}')
        
        sns.countplot(x=col, data=df, hue=TARGET, ax=axes[1], palette=colour_map)
        axes[1].set_title(f'Bar Chart of {col} by {TARGET}')
        
    plt.tight_layout()
    plt.show()

def plot_continuous_data(df: pd.DataFrame, headings, indices):

    continuous_columns = [headings[i] for i in indices]
    colour_map = sns.color_palette("rocket_r")

    for col in continuous_columns:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        sns.boxplot(x=TARGET, y=col, data=df, ax=axes[0], palette=colour_map)
        axes[0].set_title(f'Boxplot of {col}')

        sns.histplot(df[col], ax=axes[1], color=colour_map[3], kde=True, bins=30)
        axes[1].set_title(f'Distribution of {col}')

        plt.tight_layout()
        plt.show()
    
# plot_discrete_data(train_df, column_headings, discrete_columns)
# plot_continuous_data(train_df, column_headings, continuous_columns)

def generate_scatterplot(x_axis, y_axis, colour):
    plt.figure(figsize=(12,6), dpi=100)
    plt.tight_layout(pad=2.0)
    sns.scatterplot(data=train_df, x=x_axis, y=y_axis, hue="satisfaction", palette=f"blend:#000,#{colour}", alpha=0.75)
    plt.title(f"{x_axis} versus {y_axis}")
    plt.show()

# generate_scatterplot("Arrival Delay in Minutes", "Departure Delay in Minutes", "C02")
# generate_scatterplot("Age", "Departure Delay in Minutes", "C02")
# generate_scatterplot("Flight Distance", "Departure Delay in Minutes", "C02")

def variable_encoding(df):

    cleaned_df = df.dropna()
    categorical_variables = cleaned_df.select_dtypes(include="object")
    categorical_names= categorical_variables.columns

    class_levels = ["Eco","Eco Plus","Business"]
    enc = OrdinalEncoder(categories=[class_levels])
    categorical_variables["Class"] = enc.fit_transform(categorical_variables[["Class"]])
    categorical_variables["Class"] = pd.to_numeric(categorical_variables["Class"],downcast="integer")
    categorical_variables["Class"].head()

    lab_enc_vars = categorical_names.drop("Class")
    for i in lab_enc_vars:
        categorical_variables[i]=LabelEncoder().fit_transform(categorical_variables[i])

    cleaned_df[categorical_variables.columns] = categorical_variables

    return cleaned_df

encoded_train_df = variable_encoding(train_df)
encoded_test_df = variable_encoding(test_df)
encoded_validation_df = variable_encoding(validation_df)

# encoded_combined_df = pd.concat([encoded_test_df, encoded_train_df, encoded_validation_df])

def generate_heatmap(df):
    plt.figure(figsize=(14,9))
    corr_matrix = df.corr().round(2)
    sns.heatmap(corr_matrix, annot=False, cmap="rocket_r", linewidths=0.2, fmt=".2g", vmin=-0.45, vmax=1)
    plt.tight_layout(pad=5.0)
    plt.title("Pearson Correlation Heatmap")
    plt.show()

# generate_heatmap(encoded_combined_df)

def variable_standardisation(df):
    scaler = StandardScaler()
    for column in ['Age', 'Flight Distance']:
        df[column] = scaler.fit_transform(df[[column]])
    return df

standardised_train_df = variable_standardisation(encoded_train_df)
standardised_test_df = variable_standardisation(encoded_test_df)
standardised_validation_df = variable_standardisation(encoded_validation_df)

def dumping_values(df):
    rows_to_drop = df[df["Online boarding"] <= 2].index
    df.drop(rows_to_drop, inplace=True)

def dumping_features(df):
    df.drop(columns=["Gender", 
                    "Departure/Arrival time convenient", 
                    "Ease of Online booking", 
                    "Online boarding",
                    "Baggage handling",
                    "Checkin service",
                    "Gate location", 
                    "Departure Delay in Minutes", 
                    "Arrival Delay in Minutes"], axis=1, inplace=True)
    
dumping_values(standardised_train_df)
dumping_values(standardised_test_df)
dumping_values(standardised_validation_df)

dumping_features(standardised_train_df)
dumping_features(standardised_test_df)
dumping_features(standardised_validation_df)

def count_satisfaction(df):
    counter = df['satisfaction'].value_counts()
    return counter

def balance_classes(df, target_col):
    majority = df[df[target_col]==1]
    minority = df[df[target_col]==0]
    majority_downsampled = resample(majority, replace=False, n_samples=minority.shape[0], random_state=2002) 
    df_downsampled = pd.concat([majority_downsampled, minority])
    return df_downsampled

balanced_train_df = balance_classes(standardised_train_df, 'satisfaction')
satisfaction_count = count_satisfaction(balanced_train_df)

features = balanced_train_df.columns.drop(TARGET)
X = balanced_train_df[features]
Y = balanced_train_df[TARGET]
lasso = LassoCV(cv=20).fit(X,Y)

# for i in range(len(features)):
#     print(f"{features[i]}: {lasso.coef_[i]}")

model = SelectFromModel(lasso, prefit=True)
X_new = model.transform(X)
selected_features = X.columns[model.get_support()]
selected_features_list = selected_features.tolist()
selected_features_list.append('satisfaction')
selected_features_index = pd.Index(selected_features_list)

def feature_selection(df, final_selection):
    final = df[final_selection].copy()
    return final

final_train_df = feature_selection(balanced_train_df, selected_features_index)
final_test_df = feature_selection(standardised_test_df, selected_features_index)
final_validation_df = feature_selection(standardised_validation_df, selected_features_index)

final_features = final_train_df.columns.drop(TARGET)
X_train = final_train_df[final_features]
Y_train = final_train_df[TARGET]
X_validation = final_validation_df[final_features]
Y_validation = final_validation_df[TARGET]

# model = SVC(C=1, kernel='rbf', gamma=0.005)
# model = model.fit(X_train, Y_train)
# Y_predicted = model.predict(X_validation)

# acc = accuracy_score(Y_validation, Y_predicted)
# prec = precision_score(Y_validation, Y_predicted)
# rec = recall_score(Y_validation, Y_predicted)

# print(acc)
# print(prec)
# print(rec)

# cm = confusion_matrix(Y_validation, Y_predicted)
# sns.heatmap(cm, annot=True, fmt="d", cmap='rocket_r', vmin=0)
# plt.ylabel('True state')
# plt.xlabel('Satisfaction')
# plt.show()

scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def hyperparameter_tuning(x_train, y_train, metrics):

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [0.005, 0.05, 0.5, 5, 50],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    results = {}

    for i in range(len(metrics)):
    
        # svm_model = SVC()
        # grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring=metrics[i], verbose=1, n_jobs=-1)
        # grid_search.fit(x_train, y_train)
        # best_params = grid_search.best_params_
        # best_score = grid_search.best_score_
        rand_search = RandomizedSearchCV(svm_model, param_grid, cv=5, scoring=metrics[i], verbose=1, n_jobs=-1, n_iter=5)
        rand_search.fit(x_train, y_train)
        best_params = rand_search.best_params_
        best_score = rand_search.best_score_
        results[metrics[i]] = (best_params, best_score)
        print("Best parameters:", best_params)
        print("Best score:", best_score)
        # Could try courser search and then a finer search afterwards???

    return results

# ideal_tuning = hyperparameter_tuning(X_train, Y_train, scoring_metrics)
# print(ideal_tuning)

def kernel_tuning(x_train, y_train, x_test, y_test, C_test, gamma_test):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for kernel in kernels:
        model = SVC(C=C_test, kernel=kernel, gamma=gamma_test, probability=True)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probas = model.predict_proba(x_test)[:, 1]
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probas)
        
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)

    metrics = [accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    palette = sns.color_palette("rocket", len(metric_names))
    x = np.arange(len(kernels)) 
    bar_width = 0.15 

    fig, ax = plt.subplots()
    for i in range(len(metrics)):
        rects = ax.bar(x - 2*bar_width + i*bar_width, metrics[i], bar_width, label=metric_names[i], color=palette[i])

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, '{:.3f}'.format(height),
                    ha='center', va='bottom')
            
    ax.set_xlabel('Kernel')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by different kernels')
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.legend()
    plt.show()

# kernel_tuning(X_train, Y_train, X_validation, Y_validation, 1, 0.005)

def C_tuning(train_data, valid_data, test_data, kernel_test, gamma_test):
    C_values = [0.01, 0.1, 1, 10, 100]
    
    datasets = [valid_data, test_data]
    dataset_names = ['Training', 'Test']
    metrics = ['Accuracy', 'Precision', 'Recall']
    
    for metric in metrics:
        plt.figure(figsize=(10,6))
        palette = sns.color_palette("rocket", len(dataset_names))
        
        for data, name, colour in zip(datasets, dataset_names, palette):
            x_train, y_train = train_data
            x_test, y_test = data
            
            scores = []
        
            for C in C_values:
                model = SVC(C=C, kernel=kernel_test, gamma=gamma_test, probability=True)
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                
                if metric == 'Accuracy':
                    score = accuracy_score(y_test, predictions)
                elif metric == 'Precision':
                    score = precision_score(y_test, predictions)
                elif metric == 'Recall':
                    score = recall_score(y_test, predictions)
                
                scores.append(score)

            plt.plot(C_values, scores, marker='o', linestyle='-', color=colour, label=f'{name} {metric}')
            
            for i, txt in enumerate(scores):
                plt.annotate("{:.3f}".format(txt), (C_values[i], scores[i]))
                
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel(f'{metric} Score')
        plt.title(f'{metric} Scores by different C values with {kernel_test} Kernel')
        plt.legend()

    plt.show()

# C_tuning([X_train, Y_train], [X_train, Y_train], [X_validation, Y_validation], "rbf", 0.005)

def gamma_tuning(train_data, valid_data, test_data, kernel_test, C_test):
    gamma_values = [0.005, 0.05, 0.5, 5]
    
    datasets = [valid_data, test_data]
    dataset_names = ['Training', 'Test']
    metrics = ['Accuracy', 'Precision', 'Recall']
    
    for metric in metrics:
        plt.figure(figsize=(10,6))
        palette = sns.color_palette("rocket", len(dataset_names))
        
        for data, name, colour in zip(datasets, dataset_names, palette):
            x_train, y_train = train_data
            x_test, y_test = data
            
            scores = []

            sample_size = int(0.1 * len(x_train)) 
            sample_indices = np.random.choice(len(x_train), size=sample_size, replace=False)  
            x_train_sample = x_train.iloc[sample_indices] 
            y_train_sample = y_train.iloc[sample_indices]
        
            for gamma in gamma_values:
                model = SVC(C=C_test, kernel=kernel_test, gamma=gamma, probability=True)
                model.fit(x_train_sample, y_train_sample)
                predictions = model.predict(x_test)
                
                if metric == 'Accuracy':
                    score = accuracy_score(y_test, predictions)
                elif metric == 'Precision':
                    score = precision_score(y_test, predictions)
                elif metric == 'Recall':
                    score = recall_score(y_test, predictions)
                
                scores.append(score)

            plt.plot(gamma_values, scores, marker='o', linestyle='-', color=colour, label=f'{name} {metric}')
            
            for i, txt in enumerate(scores):
                plt.annotate("{:.3f}".format(txt), (gamma_values[i], scores[i]))
                
        plt.xscale('log')
        plt.xlabel('Gamma')
        plt.ylabel(f'{metric} Score')
        plt.title(f'{metric} Scores by different Gamma values with {kernel_test} Kernel and C={C_test}')
        plt.legend()

    plt.show()

# gamma_tuning([X_train, Y_train], [X_train, Y_train], [X_validation, Y_validation], "rbf", 1)

X_test = final_test_df[final_features]
Y_test = final_test_df[TARGET]

model = SVC(C=10, kernel='rbf', gamma=0.1)
model = model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)

acc = accuracy_score(Y_test, Y_predicted)
prec = precision_score(Y_test, Y_predicted)
rec = recall_score(Y_test, Y_predicted)

print(acc)
print(prec)
print(rec)

cm = confusion_matrix(Y_test, Y_predicted)
sns.heatmap(cm, annot=True, fmt="d", cmap='rocket_r', vmin=0)
plt.ylabel('True state')
plt.xlabel('Satisfaction')
plt.show()