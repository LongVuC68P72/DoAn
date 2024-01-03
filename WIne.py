import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)  # Hiển thị tất cả cột
pd.set_option('display.max_rows', None)     # Hiển thị tất cả dòng
sns.set()


df = pd.read_csv("D:\Wine.csv")
#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#----------------------------------------------
#sns.countplot(x = 'Wine',data=df)
#plt.show()
#----------------------------------------------
target = df['Wine']
df = df.drop('Wine',axis=1)
#----------------------------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df,target,test_size =0.20,random_state=42)

#sns.pairplot(X_train)
#plt.show()
#------------Implement_scaling----------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#---------------------Chuẩn hóa dữ liệu----------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#----------------------------------------------
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
#sns.pairplot(X_train)
#plt.show()
#-------------------------Xây dựng mô hình----------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(classification_report(y_test,knn.predict(X_test)))
#-------------------------Vẽ ma trận hỗn loạn chưa áp dụng PCA----------------------
#conf_matrix = confusion_matrix(y_test, knn.predict(X_test))
#plt.figure(figsize =(10,8))
#sns.heatmap(X_train.corr(),annot=True)
#plt.show()

#labels = ['1', '2', '3']

# Tạo biểu đồ heatmap cho confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
#sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
#plt.title('Confusion Matrix')
#plt.show()
#--------------------Áp dụng PCA--------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
tr_comp = pca.fit_transform(X_train)
ts_comp = pca.transform(X_test)
pc_knn = KNeighborsClassifier(n_neighbors=5)
pc_knn.fit(tr_comp,y_train)

#-------------------------Vẽ ma trận hỗn loạn áp dụng PCA----------------------
print(classification_report(y_test,pc_knn.predict(ts_comp)))
conf_matrix = confusion_matrix(y_test, pc_knn.predict(ts_comp))
labels = ['1', '2', '3']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.show()