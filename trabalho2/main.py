import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def detect_outliers_zscore(data):
    outliers = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    for idx, value in enumerate(data):
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(idx)
    return outliers


def detect_outliers_iqr(data):
    outliers = []
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    IQR = q3 - q1
    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)
    for idx, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            outliers.append(idx)
    return outliers

def cleanse_data(car_data):
    
    """ Data clean-up """

    # Learn more about the training data attributes
    print(car_data.head())
    print(car_data.describe())
    
    # Drop unuseful columns
    car_data = car_data.drop(columns=['policy_id', 'is_adjustable_steering', 'is_tpms', 'is_parking_camera', 'rear_brakes_type', 'cylinder', 'transmission_type', 'turning_radius', 'is_rear_window_washer', 'is_power_door_locks', 'is_central_locking', 'is_day_night_rear_view_mirror'])

    # Check for NULL entries - in this case, there are none, therefore no need to filter it out
    n_of_null_values = car_data.isnull().sum()
    print("Number of NULL entries in each column:\n", n_of_null_values)

    # Remove duplicate entries
    car_data = car_data.drop_duplicates()
    
    # Target encode the training data, to better suit the training models
    for column in car_data.columns:
        if dict(car_data.dtypes)[column] == 'object':  # If it is a value of type string or categorical variable     
            label_encoder = preprocessing.LabelEncoder()
            car_data[column] = label_encoder.fit_transform(car_data[column])
    

    # Remove all age values not between 0 and 1 
    subset = car_data[~car_data['age_of_policyholder'].between(0, 1)]
    print("\nNumber of entries where policy holder age not between 0 and 1: "+str(len(subset)))
    # car_data = car_data[car_data['age_of_policyholder'].between(0, 1)]

    subset = car_data[~car_data['age_of_car'].between(0, 1)]
    print("\nNumber of entries where age of car not between 0 and 1: "+str(len(subset)))
    # car_data = car_data[car_data['age_of_car'].between(0, 1)]

    subset = car_data[~car_data['policy_tenure'].between(0, 1)]
    print("\nNumber of entries where policy tenure not between 0 and 1: "+str(len(subset)))
    car_data = car_data[car_data['policy_tenure'].between(0, 1)]



    # Detect outliers using Z-Score and Inter Quartile Range
    car_data = car_data.reset_index(drop=True)

    for column in ['population_density', 'fuel_type', 'displacement', 'length','width','height','gross_weight']:
        outliers = detect_outliers_zscore(car_data[column])
        print("\nNumber of Z-score outliers in "+column+" column: "+str(len(outliers)))
        car_data.drop(outliers)

        outliers = detect_outliers_iqr(car_data[column])
        print("Number of IQR outliers in "+column+" column: "+str(len(outliers)))
        car_data.drop(outliers)

    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(car_data)
    car_data = pd.DataFrame(scaled_data, columns=car_data.columns)


    # Store transformed data
    car_data.to_csv('./data/cleansed_car_data.csv', index=False)



def main():

    car_data = pd.read_csv("data/train.csv")

    cleanse_data(car_data)

    car_data = pd.read_csv("data/cleansed_car_data.csv")
    print("\n",car_data.head())


    """ Data separation (target and feature, train and test columns) """
    
    
    # Separate the data into target and feature variables
    car_data_target = car_data['is_claim']
    car_data_features = car_data.drop('is_claim', axis=1)

    # Check for class imbalance - the results indicate that there is a clear imbalance in the target variable, an overwhelming amount of 0's compared to 1's
    print(car_data_target.value_counts(1))

    # Separate data into training and testing, provide seed for consistency
    car_data_features_train, car_data_features_test, car_data_target_train, car_data_target_test = model_selection.train_test_split(car_data_features, car_data_target, test_size = 0.3, random_state = 0)

    # Correct class imbalance using class weights. 'balanced' == n_samples / (n_classes * np.bincount(is_claim))
    weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(car_data_target_train), y = car_data_target_train)
    print(weights[0], weights[1])


    """ Machine Learning Models Training """

    # Apply K-Fold validation to each algorithm, with weights, due to imbalance (GridSearchCV / RandomizedSearchCV)

    print('-------------ALGORITHMS-------------')

    # For Decision Tree

    dtc = DecisionTreeClassifier()

    grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [1, 2, 3, 4, 5],
        'max_features': [1, 2, 3, 4]
    }

    cross_validation = StratifiedKFold(n_splits=10)

    gs_dt_cv = GridSearchCV(dtc, param_grid=grid, cv=cross_validation)
    gs_dt_cv.fit(car_data_features_train, car_data_target_train)

    print('For Decision Tree:')
    print('Best score: {}'.format(gs_dt_cv.best_score_))
    print('Best parameters: {}'.format(gs_dt_cv.best_params_))

    # For K-NN
    knc = KNeighborsClassifier()

    knn_grid = {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [10,25,50],
        'n_jobs': [10,25,50],
    }

    gs_kn_cv = GridSearchCV(knc, param_grid=knn_grid, cv=cross_validation)
    gs_kn_cv.fit(car_data_features_train, car_data_target_train)

    print('For K-NN')
    print('Best score: {}'.format(gs_kn_cv.best_score_))
    print('Best parameters: {}'.format(gs_kn_cv.best_params_))
    
    # For SVM

    # svm = SVC()

    # svc_grid = {
    #     'kernel': ['poly', 'sigmoid'],
    #     'gamma': ['scale', 'auto'],
    #     'class_weight': ['balanced'],
    # }

    # gs_svm_cv = GridSearchCV(svm, param_grid=svc_grid, cv=cross_validation)
    # gs_svm_cv.fit(car_data_features_train, car_data_target_train)

    # print('For SVM')
    # print('Best score: {}'.format(gs_svm_cv.best_score_))
    # print('Best parameters: {}'.format(gs_svm_cv.best_params_))

    # For Logistic regression

    lrc = LogisticRegression()

    lrc_grid = {
        'penalty': ['l1','l2','elasticnet'],
        'l1_ratio': [1, 0, 0.2, 0.5],
        'tol': [0.2, 0.5]
    }

    gs_lrc_cv = GridSearchCV(lrc, param_grid=lrc_grid,cv=cross_validation)
    gs_lrc_cv.fit(car_data_features_train, car_data_target_train)

    print('For Logistic Regression:')
    print('Best score: {}'.format(gs_lrc_cv.best_score_))
    print('Best parameters: {}'.format(gs_lrc_cv.best_params_))


    # TODO Measure pegsormance for each algorithm and associated weights (maybe try different ones). Visual pegsormers = Confusion Matrix, ROC for Model Comparison ; Value pegsormers = Accuracy, F-Measure

    print('-------------PERFORMANCES-------------')    

    # For Decision Tree

    print('For decision tree:')

    dtc.fit(car_data_features_train, car_data_target_train)

    dtc.score(car_data_features_train, car_data_target_train)

    dtc_pred = dtc.predict(car_data_features_train)

    cmat = confusion_matrix(car_data_target_train, dtc_pred, labels=dtc.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=dtc.classes_)

    disp.plot(cmap='gist_gray')
    disp.plot()

    plt.show()

    accuracy = accuracy_score(car_data_target_train,dtc_pred)

    precision = precision_score(car_data_target_train,dtc_pred, average="macro")

    recall = recall_score(car_data_target_train,dtc_pred, average="macro")

    f1 = f1_score(car_data_target_train,dtc_pred, average="macro")

    print(f"Accuracy: {accuracy:.5f}")

    print(f"Precision: {precision:.5f}")

    print(f"Recall: {recall:.5f}")

    print(f"F1 Accuracy: {f1:.5f}")

    #Learning Curve
    car_train_sizes, car_train_scores, car_test_scores = learning_curve(DecisionTreeClassifier(),
                    car_data_features, car_data_target, scoring = 'accuracy', n_jobs = 1)

    train_mean = np.mean(car_test_scores, axis = 1)
    train_std = np.std(car_train_scores, axis=1)

    test_mean = np.mean(car_test_scores, axis=1)
    test_std = np.std(car_test_scores, axis=1)

    plt.subplots(1, figsize=(10,10))
    plt.plot(car_train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(car_train_sizes, test_mean, color="#111111", label="Cross-validation score")

    plt.fill_between(car_train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(car_train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # For K-NN
    print('For K-NN:')

    knc.fit(car_data_features_train, car_data_target_train)

    knc.score(car_data_features_train, car_data_target_train)

    knn_pred = knc.predict(car_data_features_train)

    cmat = confusion_matrix(car_data_target_train, knn_pred, labels=knc.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=knc.classes_)

    disp.plot(cmap='gist_gray')
    disp.plot()

    plt.show()

    accuracy = accuracy_score(car_data_target_train,knn_pred)

    precision = precision_score(car_data_target_train,knn_pred, average="macro")

    recall = recall_score(car_data_target_train, knn_pred, average="macro")

    f1 = f1_score(car_data_target_train,knn_pred, average="macro")

    print(f"Accuracy: {accuracy:.5f}")

    print(f"Precision: {precision:.5f}")

    print(f"Recall: {recall:.5f}")

    print(f"F1 Accuracy: {f1:.5f}")

    # For Logisitic Regression

    print('For Logistic Regression:')

    lrc.fit(car_data_features_train, car_data_target_train)

    lrc.score(car_data_features_train, car_data_target_train)

    log_pred = lrc.predict(car_data_features_train)

    cmat = confusion_matrix(car_data_target_train, log_pred, labels=lrc.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=lrc.classes_)

    disp.plot(cmap='gist_gray')
    disp.plot()

    plt.show()

    accuracy = accuracy_score(car_data_target_train,log_pred)

    precision = precision_score(car_data_target_train,log_pred, average="macro")

    recall = recall_score(car_data_target_train, log_pred, average="macro")

    f1 = f1_score(car_data_target_train, log_pred, average="macro")

    print(f"Accuracy: {accuracy:.5f}")

    print(f"Precision: {precision:.5f}")

    print(f"Recall: {recall:.5f}")

    print(f"F1 Accuracy: {f1:.5f}")


    # TODO Train each algorithm with different sized data in order to obtain Learning Curve


if __name__ == "__main__":
    main()