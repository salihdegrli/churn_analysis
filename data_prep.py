import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, RobustScaler
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outlier_with_lof(dataframe,
                            num_cols,
                            n_neighbors_=20,
                            threshold=3,
                            plot=True,
                            inplace=False
                            ):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors_)

    lof.fit_predict(dataframe[num_cols])

    df_scores = lof.negative_outlier_factor_

    scores = pd.DataFrame(np.sort(df_scores))
    if plot:
        scores.plot(stacked=True, xlim=[0, 20], style=".-")
        plt.show()

    th = np.sort(df_scores)[threshold]

    outliers = dataframe[df_scores < th]
    if inplace:
        dataframe.drop(axis=0, labels=outliers.index, inplace=inplace)


# outlier alt ve üst sınırlarını belirler
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# değişken bazlı outlier var mı yok mu kontrol eder
def check_outlier(dataframe,
                  col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe,
                  col_name,
                  index=False):
    """
    aykırı değerlerin kendilerine erişim sağlar.
    bu işlemi de outlierlar için unique değer olan indexle sağlar

    Parameters
    ----------
    dataframe = dataframe
        üzerinde işlem yapılacak olan veri seti
    col_name = column name
        outlierın ait olduğu kolon ismi

    index = bool
        opsionel olarak index return edilmek istenirse

    Returns
    -------
    if index == true
    outlier_index

    """
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe,
                   col_name):
    """
    aykrırı değerleri drop etme işlemini gerçekleştirir

    Parameters
    ----------
    dataframe = dataframe = dataframe
        üzerinde işlem yapılacak olan veri seti
    col_name  = column name
        outlierın ait olduğu kolon ismi

    Returns
    -------
    outlierlardan arınmış dataframe
    """
    low_limit, up_limit = outlier_thresholds(dataframe,
                                             col_name)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


# aykırı değerleri sınır değerleri atar, baskılama işlemi
def replace_with_thresholds(dataframe,
                            variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def high_correlated_cols(dataframe,
                         plot=False,
                         corr_th=0.90):

    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def missing_values_table(dataframe,
                         na_name=False):
    """
    na değerlerinin olduğu değişkenleri bulur.
     bunların her değişken için kaçar tane olduğunu hesaplar.
    oran olarak dataset içerisinde %kaçını oluşturduğunu hesaplar

    na_name = bool
     na olan kolonların isimlerini ayrıca almak istersek True, aksi takdirde False

    dataframe = dataframe
     analiz yapılacak olan veri seti
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe,
                      target,
                      na_columns):

    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe,
                    categorical_cols,
                    drop_first=False):

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_encoder(dataframe,
                 rare_perc,
                 cat_cols):
    """
        1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
        rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
        eğer 1'den büyük ise rare cols listesine alınıyor.
    """

    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe


def rare_analyser(dataframe,
                  target,
                  cat_cols):

    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def missing_vis_analysis(dataframe,
                         fontsize=16,
                         labels=None):

    msno.bar(dataframe, fontsize=fontsize, labels=labels, figsize=(16, 9))
    plt.title("Missing Bar Plot")
    plt.show()
    msno.matrix(dataframe, fontsize=fontsize, labels=labels, figsize=(16, 9))
    plt.title("Missing Matrix Plot")
    plt.show()
    msno.heatmap(dataframe, fontsize=fontsize, labels=labels, figsize=(16, 9))
    plt.title("Nullity Correlation Matrix")
    plt.show()


def check_missing_value(dataframe):
    return dataframe.isnull().values.any()


def missing_at_least_one(dataframe):
    return dataframe[dataframe.isnull().any(axis=1)]


def non_missing(dataframe):
    return dataframe[dataframe.notnull().all(axis=1)]


def fillna_num_cols_in_category(num_col,
                                cat_col,
                                dataframe,
                                strategy="mean",
                                inplace=False):

    if inplace:
        dataframe[num_col] = dataframe[num_col].fillna(dataframe.groupby(cat_col)[num_col].transform(strategy))
        return dataframe
    else:
        return dataframe[num_col].fillna(dataframe.groupby(cat_col)[num_col].transform(strategy))


def fillna_cat_cols(dataframe):
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,
                                axis=0)
    return dataframe


def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def robust_scaler(dataframe, num_cols, quantile_range=(0.05, 0.95)):
    """
    Parameters
    ----------
        dataframe : dataset
        num_cols : numeric columns
        quantile_range : IQR Range (Q1 and Q3)

    Returns
    -------
        scaler : scaler object
        dataframe : scaled dataframe
    """
    scaler = RobustScaler(quantile_range=quantile_range)
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
