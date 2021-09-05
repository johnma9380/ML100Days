import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def feature_engineering_good(total: pd.DataFrame) -> pd.DataFrame:
    total_fe = total.copy()
    # remove 'Id'
    total_fe.drop(columns = ['Id'], inplace = True)

    # ------- Label Encoding -------
    # MSZoning
    map_dict = {'C (all)': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4}
    total_fe.MSZoning = total.MSZoning.map(map_dict)

    # Street
    le = LabelEncoder()
    total_fe.Street = le.fit_transform(total.Street)

    # Neighborhood
    le = LabelEncoder()
    total_fe.Neighborhood = le.fit_transform(total.Neighborhood)

    # Condition1
    total_fe.Condition1 = le.fit_transform(total.Condition1)

    # Condition2
    map_dict = {'Feedr': 0, 'RRNn': 1, 'Artery': 2, 'RRAn': 3, 'Norm': 4, 'RRAe': 5, 'PosN': 6, 'PosA': 7}
    total_fe.Condition2 = total.Condition2.map(map_dict)

    # HouseStyle
    total_fe.HouseStyle = le.fit_transform(total.HouseStyle)

    # RoofStyle
    map_dict = {'Gable': 0, 'Gambrel': 0, 'Hip': 1, 'Mansard': 1, 'Shed': 2, 'Flat': 3}
    total_fe.RoofStyle = total.RoofStyle.map(map_dict)

    # RoofMatl
    total_fe.RoofMatl = le.fit_transform(total.RoofMatl)

    # Exterior1st
    total_fe.Exterior1st = le.fit_transform(total.Exterior1st)

    # Exterior2nd
    total_fe.Exterior2nd = le.fit_transform(total.Exterior2nd)

    # MasVnrType
    map_dict = {'None': 0, 'BrkCmn': 1, 'BrkFace': 2, 'Stone': 3}
    total_fe.MasVnrType = total.MasVnrType.map(map_dict)

    # ExterQual
    map_dict = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
    total_fe.ExterQual = total.ExterQual.map(map_dict)

    # Foundation
    total_fe.Foundation = le.fit_transform(total.Foundation)

    # BsmtQual
    map_dict = {'Na': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5} 
    total_fe.BsmtQual = total.BsmtQual.map(map_dict)

    # BsmtCond
    map_dict = {'Po': 0, 'Na': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}
    total_fe.BsmtCond = total.BsmtCond.map(map_dict)

    # BsmtExposure
    map_dict = {'Na': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    total_fe.BsmtExposure = total.BsmtExposure.map(map_dict)

    # BsmtFinType1
    total_fe.BsmtFinType1 = le.fit_transform(total.BsmtFinType1)

    # BsmtFinType2
    total_fe.BsmtFinType2 = le.fit_transform(total.BsmtFinType2)

    # Heating
    total_fe.Heating = le.fit_transform(total.Heating)

    # HeatingQC
    total_fe.HeatingQC = le.fit_transform(total.HeatingQC)

    # CentralAir
    map_dict = {'N': 0, 'Y': 1}
    total_fe.CentralAir = total.CentralAir.map(map_dict)

    # Electrical
    total_fe.Electrical = le.fit_transform(total.Electrical)

    # KitchenQual
    map_dict = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
    total_fe.KitchenQual = total.KitchenQual.map(map_dict)

    # Functional
    total_fe.Functional = le.fit_transform(total.Functional)

    # FireplaceQu
    total_fe.FireplaceQu = le.fit_transform(total.FireplaceQu)

    # GarageType
    total_fe.GarageType = le.fit_transform(total.GarageType)

    # GarageFinish
    map_dict = {'Na': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    total_fe.GarageFinish = total.GarageFinish.map(map_dict)

    # GarageQual
    total_fe.GarageQual = le.fit_transform(total.GarageQual)

    # GarageCond
    total_fe.GarageCond = le.fit_transform(total.GarageCond)

    # PavedDrive
    map_dict = {'N': 0, 'P': 1, 'Y': 2}
    total_fe.PavedDrive = total.PavedDrive.map(map_dict)

    # MiscFeature
    # total_fe.MiscFeature = le.fit_transform(total.MiscFeature)

    # SaleType
    total_fe.SaleType = le.fit_transform(total.SaleType)

    # SaleCondition
    total_fe.SaleCondition = le.fit_transform(total.SaleCondition)

    # ------- One-hot Encoding -------
    discrete = [f for f in total_fe.columns if total_fe.dtypes[f] == 'object']
    for feature in discrete:
        dummies = pd.get_dummies(total_fe[feature], prefix = feature)
        total_fe = pd.concat([total_fe, dummies], axis = 1)
        total_fe.drop(columns = [feature], inplace = True)

    # ------- feature combination -------
    total_fe['YrBltAndRemod'] = total_fe['YearBuilt'] + total_fe['YearRemodAdd']
    total_fe['TotalSF'] = total_fe['TotalBsmtSF'] + total_fe['1stFlrSF'] + total_fe['2ndFlrSF']

    total_fe['Total_sqr_footage'] = (
        total_fe['BsmtFinSF1'] + total_fe['BsmtFinSF2'] +
        total_fe['1stFlrSF'] + total_fe['2ndFlrSF']
    )

    total_fe['Total_Bathrooms'] = (
        total_fe['FullBath'] + (0.5 * total_fe['HalfBath']) +
        total_fe['BsmtFullBath'] + (0.5 * total_fe['BsmtHalfBath'])
    )

    total_fe['Total_porch_sf'] = (
        total_fe['OpenPorchSF'] + total_fe['3SsnPorch'] +
        total_fe['EnclosedPorch'] + total_fe['ScreenPorch'] +
        total_fe['WoodDeckSF']
    )

    drop_columns = [
        'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'FullBath',
        'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
        'OpenPorchSF', '3SsnPorch', 'EnclosedPorch',
        'ScreenPorch', 'WoodDeckSF'
    ]

    total_fe.drop(columns = drop_columns, inplace = True)

    return total_fe




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
    

