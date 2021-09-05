import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def view_miss_data(total):
    missing = total.isnull().sum(axis = 0)
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)

    plt.figure(figsize = (8, 4))
    plt.bar(list(missing.index), np.array(missing))
    plt.xticks(rotation='vertical')
    plt.show()
    print(missing)



def view_discrete_data(total, feature_name, data_y, column_name_y):
    total[column_name_y] = data_y
    df = pd.DataFrame()
    df['mean'] = total.groupby(feature_name)[column_name_y].mean()
    df['max'] = total.groupby(feature_name)[column_name_y].max()
    df['min'] = total.groupby(feature_name)[column_name_y].min()
    df['size'] = total.groupby(feature_name)[column_name_y].size()
    print(df)

    for i in total[feature_name].unique():
        # 繪製 Empirical Cumulative Density Plot (ECDF)
        cdf = total[total[feature_name] == i].sort_values(by=column_name_y,ascending=True).reset_index(drop=True)
        plt.plot(list(cdf.index), cdf[column_name_y], '-o', label = i)

    plt.xlabel('rows')
    plt.ylabel(column_name_y)
    plt.title("CFD-value& and rows")
    plt.legend()
    plt.show()

    # 繪製 Empirical Cumulative Density Plot (ECDF)
    cdf = total.sort_values(by=feature_name,ascending=True).reset_index(drop=True)
    plt.plot(cdf[feature_name], cdf[column_name_y], 'o', label = i)
    plt.xlabel(feature_name)
    plt.ylabel(column_name_y)
    plt.title("X - Y")
    plt.legend()
    plt.show()

    plt.figure(figsize = (30, 10))
    plt.subplot(1, 1, 1)
    sns.boxplot(cdf[feature_name], cdf[column_name_y])
    plt.xlabel(feature_name)
    plt.ylabel(column_name_y)
    plt.title("IQR")
    plt.show()

    total.drop([column_name_y] , axis=1)




def view_continual_data(total, feature_name, data_y, column_name_y):
    total[column_name_y] = data_y
    
    cdf = total.sort_values(by=feature_name,ascending=True).reset_index(drop=True)
    plt.plot(cdf.index, cdf[feature_name], '-')
    plt.xlabel('rows')
    plt.ylabel(feature_name)
    plt.title("CFD-value& and rows")
    plt.legend()
    plt.show()


    
    plt.plot(cdf[feature_name], cdf[column_name_y], 'o')
    plt.xlabel(feature_name)
    plt.ylabel(column_name_y)
    plt.title("X - Y")
    plt.legend()
    plt.show()


    plt.plot(cdf[feature_name], cdf[column_name_y], '-o')
    plt.xlabel(feature_name)
    plt.ylabel(column_name_y)
    plt.title("X - Y")
    plt.legend()
    plt.show()

    import seaborn as sns
    mean_df = total[feature_name].fillna(total[feature_name].mean())  
    median_df = total[feature_name].fillna(total[feature_name].median())

    label = ['Origin', 'Mean', 'Median']
    color = ['tab:blue', 'tab:orange', 'tab:green']
    datas = [total[feature_name], mean_df, median_df]
    plt.figure(figsize = (20, 5), dpi = 80)
    plt.subplot(1, 3, 1)
    for i in range(3):
        sns.histplot(datas[i], kde = True, stat = 'probability', element = 'bars', label = label[i], color = color[i])
    
    plt.title("Origin - Mean - Median")
    plt.legend()
    plt.show()

    print(total[feature_name].mean(), total[feature_name].max(), total[feature_name].min())


    import math
    def cut_bin_by_interval(df, col_name, val):
        bin_cnt = math.ceil( (df[col_name].max() - df[col_name].min()) / val) + 1
        bins_list = [0 - val] + [ i*val for i in range(bin_cnt)]
        score_cut = pd.cut(df[col_name], bins_list, labels=False, retbins=True, right=True)
        score_list = []
        for i in score_cut[0]:
            score_list.append(score_cut[1][int(i)])
        return score_list

    total[f'{feature_name}_Categories'] =  cut_bin_by_interval(total, feature_name, 100)
    # 繪製 Empirical Cumulative Density Plot (ECDF)
    cdf = total.sort_values(by=feature_name,ascending=True).reset_index(drop=True)
    plt.plot(cdf[f'{feature_name}_Categories'], cdf[column_name_y], 'o', label = i)
    plt.xlabel(f'{feature_name}_Categories')
    plt.ylabel(column_name_y)
    plt.title("Binging")
    plt.legend()
    plt.show()


    plt.figure(figsize = (30, 10))
    plt.subplot(1, 1, 1)
    sns.boxplot(cdf[f'{feature_name}_Categories'], cdf[column_name_y]/cdf[column_name_y].max())
    # plt.title(label = feature, fontdict = {'fontsize': 10})
    plt.xlabel(f'{feature_name}_Categories')
    plt.ylabel(column_name_y)
    plt.title("Binging-IQR")
    plt.show()


    total.drop([column_name_y] , axis=1)