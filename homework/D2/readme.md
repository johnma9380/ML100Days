# Hw02
## 解題步驟：
1. 取出連續型特徵，以便供後續演算法降維用
2. 分析缺失值，練習繪製直方圖
3. 去除缺失值
4. 將特徵進行標準化，並將目標 SalePrice 透過四分位距分成四類
5. 利用 TSNE 演算法進行降至二維
6. 繪製散佈圖 (如下圖)
7. 課後問答

<img src="./tsne_cluster.png" style="zoom:90%;" />

## Hint

* 去除缺失值請參考課堂一的投影片 [(dropna)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html)
* 透過四分位距切分成四類 [(pd,qcut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html))
* [TSNE 演算法詳細教學](https://mropengate.blogspot.com/2019/06/t-sne.html)
* [TSNE Sklearn 使用教學](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
* 在繪製資料點分佈圖時可以使用 [plt.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) 來繪製，或使用更方便的 [sns.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)，可以繪製更美觀的圖表

## 檔案解說：

1. Matplotlib_Seaborn.ipynb:
   * 紀錄上課關於 Matplotlib, Seaborn 的教學
2. data_analysis.ipynb:
   * 紀錄上課關於資料分析的教學
3. demo.ipynb
   * 上課示範用程式碼
4. HW_02.ipynb:
   * 同學需撰寫的程式作業
5. HW_02_ans.ipynb:
   * 作業的正確答案
6. data_description.md
   * 關於特徵的講解
7. train.csv:
   * 同學寫作業時須讀取的檔案
8. tsne_cluster.png:
   * readme.md 中的參考圖片