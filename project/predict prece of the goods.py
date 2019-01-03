from sklearn.linear_model import LinearRegression   #线性回归
from sklearn.tree import DecisionTreeRegressor      #决策树回归

#支持向量回归
from sklearn.svm import SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,Y_train.ravel())
linear_svr.predict = linear_svr.predict(X_test)

#梯度提升回归
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
gbdt = GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    min_samples_leaf=1,
    min_samples_split=2,
    max_depth=3,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False
)
train_feat = np.genfromtxt('train_feat',dtype=np.float32)
train_id=np.genfromtxt("train_id.txt",dtype=np.float32)
test_feat=np.genfromtxt("test_feat.txt",dtype=np.float32)
test_id=np.genfromtxt("test_id.txt",dtype=np.float32)
gbdt.fit(train_feat,train_id)
pred = gbdt.predict(test_feat)
total_err = 0
for i in range(pred.shape[0]):
    print(pred[i],test_id[i])
    err = (pred[i]-test_id[i])/train_id[i]
    total_err += err * err
print(total_err/pred.shape[0])

#xgboost线性回归
import xgboost as xgb
regr = xgb.XGBRegressor()
