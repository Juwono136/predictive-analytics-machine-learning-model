import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# load the dataset
dataset = 'house-prices/train.csv'
house = pd.read_csv(dataset)
house
house.info()

# cek nilai Null/NaN pada nilai fitur
nan_counts = house.isna().sum()
columns_with_nan = nan_counts[nan_counts > 0].index.tolist()

nan_info = pd.DataFrame({
    'Fitur': columns_with_nan,
    'Jumlah NaN/Null': nan_counts[columns_with_nan].tolist()
})

print("Informasi NaN/Null dalam bentuk DataFrame:")
print(nan_info)

columns_with_nan = house.columns[house.isna().any()].tolist()

print("Fitur yang mengandung NaN:")
print(columns_with_nan)

# drop fitur yang mengandung NaN/Null
house.drop(['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
           'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)

house

house.describe()

# Identifikasi Missing Value
# Melihat informasi semua fitur dengan nilai minimum 0
describe_result = house.describe()
min_values_0 = (describe_result.loc["min"] == 0)
col_with_min_0 = min_values_0[min_values_0].index

print(f"Jumlah kolom dengan nilai minimum 0: {len(col_with_min_0)}\n")
print("Fitur dengan nilai minimum 0:")
print(f"{col_with_min_0}\n")

for min_0 in col_with_min_0:
    index_min_0 = (house[min_0] == 0).sum()
    print(f"Nilai 0 di kolom {min_0} ada: ", index_min_0)

house.shape

# Menghilangkan outliers
# fitur MSSubClass
sns.boxplot(x=house['MSSubClass'])

# fitur LotArea
sns.boxplot(x=house['LotArea'])

# metode IQR
Q1 = house.quantile(0.25)
Q3 = house.quantile(0.75)
IQR = Q3-Q1
house = house[~((house < (Q1-1.5*IQR)) | (house > (Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
house.shape

# cek kembali fitur MSSubClass dari outliers
sns.boxplot(x=house['MSSubClass'])

# Menghapus fitur yang tidak diperlukan
# Menghapus kolom Id
house = house.drop('Id', axis=1)
house.head()

for column in house.columns:
    if len(house[column].unique()) == 1:
        print(f"Fitur '{column}' memiliki hanya satu nilai unik.")

for column in house.columns:
    if len(house[column].unique()) == 1:
        house.drop(column, axis=1, inplace=True)

house.head()

# Exploratory Data Analysis - Univariate Analysis
numerical_features = house.select_dtypes(
    include=['float64', 'int64']).columns.tolist()
categorical_features = house.select_dtypes(include=['object']).columns.tolist()

# categorical features
for feature in categorical_features:
    count = house[feature].value_counts()
    percent = 100 * house[feature].value_counts(normalize=True)
    df = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})
    print(f"Statistik untuk fitur {feature}:\n")
    print(df)
    print("\n")

    # Visualisasi dengan bar plot
    plt.figure(figsize=(15, 5))
    count.plot(kind='bar', title=feature)
    plt.show()

# numerical features
house.hist(bins=50, figsize=(20, 15))
plt.show()

# Exploratory Data Analysis - Multivariate Analysis
# Categorical Features
cat_features = house.select_dtypes(include='object').columns.to_list()

for col in cat_features:
    sns.catplot(x=col, y="SalePrice", kind="bar", dodge=False,
                height=4, aspect=3,  data=house, palette="Set3")
    plt.xticks(rotation=45)
    plt.title("Rata-rata 'SalePrice' Relatif terhadap - {}".format(col))

# numerical features
plt.figure(figsize=(20, 10))
correlation_matrix = house.corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True,
            cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

house.drop(["MSSubClass", "LotArea", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
           "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces", "WoodDeckSF", "OpenPorchSF", "MoSold", "YrSold"], inplace=True, axis=1)

house.head()

numerical_features = house.select_dtypes(include=['float64', 'int64'])
numerical_features = numerical_features.dropna()
numerical_features

# Data Preparation
# Encoding fitur kategori
cat_features = house.select_dtypes(include='object').columns.to_list()

for feature in cat_features:
    dummies = pd.get_dummies(house[feature], prefix=feature)
    house = pd.concat([house, dummies], axis=1)

house.drop(cat_features, axis=1, inplace=True)
house.head()

# Reduksi Dimensi dengan PCA
numerical_features = house.select_dtypes(include=['float64', 'int64'])

sns.pairplot(house[numerical_features.columns.tolist()], plot_kws={"s": 3})

sns.pairplot(house[['GrLivArea', 'GarageArea']], plot_kws={"s": 3})

pca = PCA(n_components=2, random_state=123)
pca.fit(house[['GrLivArea', 'GarageArea']])
princ_comp = pca.fit_transform(house[['GrLivArea', 'GarageArea']])

pca.explained_variance_ratio_.round(3)

pca = PCA(n_components=1, random_state=123)
pca.fit(house[['GrLivArea', 'GarageArea']])
house['dimension'] = pca.fit_transform(
    house.loc[:, ('GrLivArea', 'GarageArea')]).flatten()
house.drop(['GrLivArea', 'GarageArea'], axis=1, inplace=True)

# Train-Test-Split
X = house.drop(["SalePrice"], axis=1)
y = house["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# Standarisasi
numerical_features = ['OverallQual', 'GarageCars', 'dimension']

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.fit_transform(
    X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

# Modeling
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

# KNN
model_knn = KNeighborsRegressor()
parameters_knn = {
    'n_neighbors': [10, 20, 30, 40, 50, 60, 70, 80]
}

grid_search_knn = GridSearchCV(
    model_knn, parameters_knn, scoring='neg_mean_squared_error', cv=5)

grid_search_knn.fit(X_train, y_train)
print("KNN GridSearch score: "+str(grid_search_knn.best_score_))
print("KNN GridSearch params: ")
print(grid_search_knn.best_params_)


knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse', 'knn'] = mean_squared_error(
    y_pred=knn.predict(X_train), y_true=y_train)

# Random forest
parameters_RF = {
    'n_estimators': [30, 40, 50, 60, 70, 80],
    'max_depth': [16, 32, 64, 128]
}

grid_search_RF = GridSearchCV(RandomForestRegressor(
    random_state=55, n_jobs=-1), parameters_RF, scoring='neg_mean_squared_error', cv=5)

grid_search_RF.fit(X_train, y_train)
print("RF GridSearch score: "+str(grid_search_RF.best_score_))
print("RF GridSearch params: ")
print(grid_search_RF.best_params_)

# buat model prediksi
RF = RandomForestRegressor(
    n_estimators=60, max_depth=32, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse', 'RandomForest'] = mean_squared_error(
    y_pred=RF.predict(X_train), y_true=y_train)

# Boosting algorithm
parameters_boost = {
    'learning_rate': [0.1, 0.01, 0.05],
}

grid_search_boost = GridSearchCV(AdaBoostRegressor(
    random_state=55), parameters_boost, scoring='neg_mean_squared_error', cv=5)

grid_search_boost.fit(X_train, y_train)
print("Boosting GridSearch score: "+str(grid_search_boost.best_score_))
print("Boosting GridSearch params: ")
print(grid_search_boost.best_params_)

boosting = AdaBoostRegressor(learning_rate=0.1, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse', 'Boosting'] = mean_squared_error(
    y_pred=boosting.predict(X_train), y_true=y_train)

# Evaluation
# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(
    X_test[numerical_features])

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(
        y_true=y_train, y_pred=model.predict(X_train))/1e6
    mse.loc[name, 'test'] = mean_squared_error(
        y_true=y_test, y_pred=model.predict(X_test))/1e6

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

# prediksi
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true': y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
