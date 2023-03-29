import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_data(data):
    pd.set_option('display.max_columns', None)
    data_types = data.dtypes
    value_counts = data.apply(lambda x: x.unique())
    unique_counts = data.apply(lambda x: x.nunique())
    missing_percentages = data.isnull().mean() * 100
    result = pd.concat([data_types, value_counts, unique_counts,
                        missing_percentages], axis=1)
    result.columns = ['Data Type', 'Possible Values',
                      'Unique Values','Missing Values(%)']
    return result

# 打印各列数据出现频数
def plot_value_counts(data):
    for col in data.columns:
        value_counts = data[col].value_counts()
        print(col,'\n',value_counts,'\n')

# 计算浮点数类型的5数概括
def summarize_numerical_attribute(data, attribute_name):
    #data：数据集，attribute_name:指定列名
    # 利用describe函数计算5数概括的值
    attribute_stats = data[attribute_name].describe()
    # 统计该列missing值的个数
    missing_count = data[attribute_name].isnull().sum()
    # 将获得的结果放到字典中，方便后续输出
    summary_dict = {
        'count': attribute_stats['count'],
        'mean': attribute_stats['mean'],
        'std': attribute_stats['std'],
        'min': attribute_stats['min'],
        '25%': attribute_stats['25%'],
        '50%': attribute_stats['50%'],
        '75%': attribute_stats['75%'],
        'max': attribute_stats['max'],
        'missing_count': missing_count
        }
    # 打印5数概括和缺失值个数
    print(f"Name: ",attribute_name,'\n'
          f"Count: {summary_dict['count']}\n"
          f"Mean: {summary_dict['mean']}\n"
          f"Std: {summary_dict['std']}\n"
          f"Min: {summary_dict['min']}\n"
          f"25%: {summary_dict['25%']}\n"
          f"50%: {summary_dict['50%']}\n"
          f"75%: {summary_dict['75%']}\n"
          f"Max: {summary_dict['max']}\n"
          f"Missing count: {summary_dict['missing_count']}")
    return
# 指定列画出直方图
def histogram(data,column_names):
    for column_name in column_names:
        sns.histplot(data=data, x = column_name)
        plt.show()
# 指定列画出盒图
def boxplot(data,column_names):
    for column_name in column_names:
        sns.boxplot(data=data,x=column_name)
        plt.show()

def drop_missing_rows(data):
    #检查数据集中是否有缺失值
    if data.isnull().values.any():
        return data.dropna(axis=0)
    else:
        return data

def fill_by_frenquent(data):
    for column in data.columns:
        mode_value = data[column].mode()[0]
        data[column] = data[column].fillna(mode_value)
    print(data)
    return data

#利用相关度来填充缺失值
def fill_by_corr(data,column_name):
    corr_matrix = data.corr()
    corr_series = corr_matrix[column_name]

    #找到与指定列高度相关的属性
    correlated_columns = corr_series[abs(corr_series) > 0.5].index.tolist()
    #去掉指定列本身
    correlated_columns.remove(column_name)
    #若没有与指定列高度相关的属性，则返回原始数据集
    if not correlated_columns:
        print(1)
        return data

    #使用高度相关的属性来填充缺失值
    for correlated_column in correlated_columns:
        #找到没有缺失值的数据
        non_missing_data = data[data[correlated_column].notnull()]

        #计算相关系数
        corr_coef = np.corrcoef(non_missing_data[correlated_column],
                                non_missing_data[column_name])[0][1]

        #如果相关系数为nan，则跳过该属性
        data.loc[data[column_name].isnull(),column_name] =(data.loc[data[column_name].isnull(),
                                                                    correlated_column] * corr_coef)
    return data

def fill_by_similarity(data, column):
    # 找到与指定列相似的属性
    similar_columns = data.corr()[column].sort_values(ascending=False).index.tolist()

    # 去掉指定列本身
    similar_columns.remove(column)

    # 如果没有与指定列相似的属性，则返回原始数据集
    if not similar_columns:
        return data

    # 使用相似属性来填充缺失值
    for similar_column in similar_columns:
        # 找到没有缺失值的数据
        non_missing_data = data[data[similar_column].notnull()]

        # 计算数据对象之间的相似性
        similarities = non_missing_data.apply(lambda x: np.corrcoef(non_missing_data[similar_column], x)[0, 1], axis=0)

        # 选择相似性最高的数据对象
        most_similar_index = similarities.drop(similar_column).sort_values(ascending=False).index[0]

        # 使用相似的数据对象来填充缺失值
        data.loc[data[column].isnull(), column] = data.loc[data[column].isnull(), most_similar_index]

    return data

Tweet = pd.read_csv('D:\上课资料\数据挖掘\第四周数据集\评论\删减版.csv',index_col=0)
Tweet = Tweet.reset_index(drop=True)
print(Tweet.columns)
print('数据集尺寸为：',Tweet.shape)
print(show_data(Tweet))
plot_value_counts(Tweet)
print(summarize_numerical_attribute(Tweet,'LAST_PRICE'),'\n')
print(summarize_numerical_attribute(Tweet,'1_DAY_RETURN'),'\n')
print(summarize_numerical_attribute(Tweet,'2_DAY_RETURN'),'\n')
print(summarize_numerical_attribute(Tweet,'3_DAY_RETURN'),'\n')
print(summarize_numerical_attribute(Tweet,'7_DAY_RETURN'),'\n')
print(summarize_numerical_attribute(Tweet,'PX_VOLUME'),'\n')
print(summarize_numerical_attribute(Tweet,'VOLATILITY_10D'),'\n')
print(summarize_numerical_attribute(Tweet,'VOLATILITY_30D'),'\n')

his_box_column = Tweet[['LAST_PRICE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN',
                '7_DAY_RETURN', 'PX_VOLUME', 'VOLATILITY_10D', 'VOLATILITY_30D']]
print(his_box_column.head())
#histogram(Tweet,his_box_column)
#boxplot(Tweet,his_box_column)