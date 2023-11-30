from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def normalization_options(data, method):
    print("Applying Normalization Technique....\n")
    if method == 'Standard':
        scaler = StandardScaler()
    elif method == 'MinMax':
        scaler = MinMaxScaler()
    elif method == 'Robust':
        scaler = RobustScaler()

    
    normalized_data = scaler.fit_transform(data)
    print(f"Applied {scaler} normalization Technique.")
    return normalized_data

