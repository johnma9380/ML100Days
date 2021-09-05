import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def feature_engineering(total: pd.DataFrame) -> pd.DataFrame:
    int_features = []
    float_features = []
    object_features = []

    for dtype, feature in zip(total.dtypes, total.columns):
        if dtype == 'float64':
            float_features.append(feature)
        elif dtype == 'int64':
            int_features.append(feature)
        else: # dtype == 'object':
            object_features.append(feature)

    # 老師的特徵工程
    # MSZoning
    # map_dict = {'C (all)': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4}
    # total.MSZoning = total.MSZoning.map(map_dict)
    # # Condition2
    # map_dict = {'Feedr': 0, 'RRNn': 1, 'Artery': 2, 'RRAn': 3, 'Norm': 4, 'RRAe': 5, 'PosN': 6, 'PosA': 7}
    # total.Condition2 = total.Condition2.map(map_dict)
    # # RoofStyle
    # map_dict = {'Gable': 0, 'Gambrel': 0, 'Hip': 1, 'Mansard': 1, 'Shed': 2, 'Flat': 3}
    # total.RoofStyle = total.RoofStyle.map(map_dict)
    # # MasVnrType
    # map_dict = {'None': 0, 'BrkCmn': 1, 'BrkFace': 2, 'Stone': 3}
    # total.MasVnrType = total.MasVnrType.map(map_dict)
    # # CentralAir
    # map_dict = {'N': 0, 'Y': 1}
    # total.CentralAir = total.CentralAir.map(map_dict)
    # # PavedDrive
    # map_dict = {'N': 0, 'P': 1, 'Y': 2}
    # total.PavedDrive = total.PavedDrive.map(map_dict)




    # 我的特徵工程
    for i in object_features:
        if 'TA' in total[i].unique():
            map_dict = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Na':0}
            total[i] = total[i].map(map_dict)

    for i in object_features:
        if 'GLQ' in total[i].unique():
            map_dict = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'Na':0}
            total[i] = total[i].map(map_dict)

    for i in object_features:
        if 'Fin' in total[i].unique():
            map_dict = {'Fin':3,'RFn':2,'Unf':1,'Na':0}
            total[i] = total[i].map(map_dict)

    for i in object_features:
        if 'IR2' in total[i].unique():
            map_dict = {'IR3':4,'IR2':3,'IR1':2,'Reg':1}
            total[i] = total[i].map(map_dict)

    for i in object_features:
        if 'Gtl' in total[i].unique():
            map_dict = {'Sev':3,'Mod':2,'Gtl':1}
            total[i] = total[i].map(map_dict)

    for i in object_features:
        if 'Av' in total[i].unique():
            map_dict = {'Gd':4,'Av':3,'Mn':2,'No':1,'Na':0}
            total[i] = total[i].map(map_dict)
    
    # ------- One-hot Encoding -------
    last_object_features = [f for f in total.columns if total.dtypes[f] == 'object']
    for feature in last_object_features:
        dummies = pd.get_dummies(total[feature], prefix = feature)
        total = pd.concat([total, dummies], axis = 1)
        total.drop(columns = [feature], inplace = True)

    total['YrBltAndRemod'] = total['YearBuilt'] + total['YearRemodAdd']
    total['TotalSF'] = total['TotalBsmtSF'] + total['1stFlrSF'] + total['2ndFlrSF']

    total['Total_sqr_footage'] = (
        total['BsmtFinSF1'] + total['BsmtFinSF2'] +
        total['1stFlrSF'] + total['2ndFlrSF']
    )

    total['Total_Bathrooms'] = (
        total['FullBath'] + (0.5 * total['HalfBath']) +
        total['BsmtFullBath'] + (0.5 * total['BsmtHalfBath'])
    )

    total['Total_porch_sf'] = (
        total['OpenPorchSF'] + total['3SsnPorch'] +
        total['EnclosedPorch'] + total['ScreenPorch'] +
        total['WoodDeckSF']
    )

    drop_columns = [
        'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'FullBath',
        'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
        'OpenPorchSF', '3SsnPorch', 'EnclosedPorch',
        'ScreenPorch', 'WoodDeckSF'
    ]

    total.drop(columns = drop_columns, inplace = True)

    return total




def k_means_binning(total, column_name, k):
    km = KMeans(n_clusters=10)
    y_pred = km.fit_predict(total[[column_name]])
    c = pd.DataFrame(km.cluster_centers_).sort_values(0)
    w = c.rolling(2).mean().iloc[1:] 
    w = [ total[column_name].min()-1] + list(w[0]) + [total[column_name].max()+1 ] 
    data_kmean = pd.cut(total[column_name], w, labels=False, retbins=True)

    ld = []
    for i in data_kmean[0]:
        ld.append(data_kmean[1][i])
    
    return ld
    

