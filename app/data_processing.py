from sklearn.preprocessing import StandardScaler

def Drop_manq(data):
    data.fillna(data.median(), inplace=True)
    
    return data

def Norm_data(data):
    numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data