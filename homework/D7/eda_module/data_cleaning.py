# include module
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# the function of data cleaning 
def data_cleaning(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # 將 training data 與 testing data 合併
    total = pd.concat([train, test], axis = 0)

    total = total.drop(['PoolQC'] , axis=1)
    # total = total.drop(['MiscFeature'] , axis=1)   # 先保留這個要測測看
    total['MiscFeature'] = total['MiscFeature'].fillna('Na')  
    # total['PoolQC'] = total['PoolQC'].fillna('Na')  
    total['Alley'] = total['Alley'].fillna('Na')  
    total['Fence'] = total['Fence'].fillna('Na')  
    total['FireplaceQu'] = total['FireplaceQu'].fillna('Na') 
    # 有確認過跟遮庫相關的是真的沒有車庫所以是NA
    total['GarageCond'] = total['GarageCond'].fillna('Na')  
    total['GarageQual'] = total['GarageQual'].fillna('Na')  
    total['GarageFinish'] = total['GarageFinish'].fillna('Na')  
    total['GarageType'] = total['GarageType'].fillna('Na')  
    total['BsmtFinType2'] = total['BsmtFinType2'].fillna('Na') 
    # 因為有確認到BsmtFinType2有一筆資料是真的沒有NA的資料，所以填入眾數
    from scipy.stats import mode
    mode_get = mode(total[~total['BsmtFinType2'].isnull()]['BsmtFinType2'])
    total.loc[58, 'BsmtFinType2'] = mode_get[0][0]

    total['BsmtExposure'] = total['BsmtExposure'].fillna('Na')
    total['BsmtFinType1'] = total['BsmtFinType1'].fillna('Na')
    total['BsmtCond'] = total['BsmtCond'].fillna('Na')
    total['BsmtQual'] = total['BsmtQual'].fillna('Na')

    total['MasVnrType'] = total['MasVnrType'].fillna('None')
    from scipy.stats import mode
    mode_get = mode(total[~total['Electrical'].isnull()]['Electrical'])
    total['Electrical'] = total['Electrical'].fillna(mode_get[0][0])

    # 連續型
    total['LotFrontage'] = total['LotFrontage'].fillna(total['LotFrontage'].mean())  
    total['GarageYrBlt'] = total['GarageYrBlt'].fillna(2022)  
    total['MasVnrArea'] = total['MasVnrArea'].fillna(0)  

    return total