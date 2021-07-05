# Hw03
## 解題步驟：
1. 讀取 train.csv, test.csv 並將其合併
2. 取出 LotFrontage, GarageYrBlt, MasVnrArea 三個特徵的資料
3. 分別以平均值及中位數來填補缺失值
4. 分析缺失值，練習繪製直方圖及機率圖 (如下圖)
5. 分別計算原本、以平均值填補、以中位數填補的標準差
6. 課後問答

<img src="./data/hist_plot.png" style="zoom:100%;" />

## Hint

* [pandas.DataFrame.fillna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html)
* [seaborn.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html)

## 檔案解說：

1. data_cleaning.ipynb:
   * 紀錄上課關於資料清洗的教學
4. HW_03.ipynb:
   * 同學需撰寫的程式作業
5. HW_03_ans.ipynb:
   * 作業的正確答案
6. data_description.md
   * 關於特徵的講解
7. ./data/train.csv:
   * 同學寫作業時須讀取的檔案
6. ./data/test.csv:
   * 同學寫作業時須讀取的檔案
7. ./data/hist_plot.png:
   * readme.md 中的參考圖片