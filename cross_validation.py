from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

def cross_validation_options(model, data, target, cv_method, k=5):
    print("Applying Cross-Validation Technique....\n")
    if cv_method == 'kfold':
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
    elif cv_method == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    elif cv_method == 'leave_one_out':
        cv = LeaveOneOut()

    # Performance metrics for different k values
    scores_accuracy = cross_val_score(model, data, target, cv=cv, scoring='accuracy')
    scores_precision = cross_val_score(model, data, target, cv=cv, scoring='precision_macro')
    scores_recall = cross_val_score(model, data, target, cv=cv, scoring='recall_macro')
    scores_f1 = cross_val_score(model, data, target, cv=cv, scoring='f1_macro')

    # Applying this for getting the best model after applying the cross validation based on test score(test_r2)
    cv_results = cross_validate(model, data, target, cv=cv, return_estimator=True, scoring='accuracy')
    trained_models = cv_results['estimator']
    scores = cv_results['test_score']
    best_model_index = scores.argmax()
    # Best Cross-Validated Model
    best_model = trained_models[best_model_index]
    print(f"Applied {cv} cross validation technique")
    return scores_accuracy, scores_precision, scores_recall, scores_f1, best_model