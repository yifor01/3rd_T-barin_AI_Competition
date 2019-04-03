# 2.LightGBM 
import winsound
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

ldata = lgb.Dataset(data,label=y)

def LGBM_param_selection(X,y, nfolds):
#    param = {'n_estimators':[600,900,1200,2400,2800],
#            'learning_rate':[0.005,0.00764107,0.010,0.020,0.030,0.040,0.050,0.060,0.070,
#                             0.080,0.090,0.1,0.15,0.2],
#            'max_depth':[-1,1,2,3,4,5,6,7,8,9,10,11],
#            'min_child_weight':[0,0.001,1,2,3,4,5],
#            'metric': 'mae' }
#
#            'colsample_bytree':[0.7,0.8,0.9,1],
#            'subsample':[0.7,0.8,0.9,1] ,
    param = {'n_estimators':[2800],
            'learning_rate':[0.005,0.01,0.02],
            'max_depth':[7,8,9,10],
            'min_child_weight':[0],
            'colsample_bytree':[0.4,0.6,0.8],
            'subsample':[0.4,0.6] 
            }
    
    grid_search = GridSearchCV(estimator =lgb.LGBMRegressor(), 
                               param_grid = param, scoring='neg_mean_absolute_error',
                               cv=nfolds, verbose = 2,n_jobs=3)
    grid_search.fit(X,y)
    grid_search.best_params_
    return grid_search.best_params_


LGBM_param_selection(data,y, 5 )


#{'colsample_bytree': 0.6,
# 'learning_rate': 0.01,
# 'max_depth': 6,
# 'min_child_weight': 0,
# 'n_estimators': 2800,
# 'subsample': 0.6}




model_LGB = lgb.LGBMRegressor(n_estimators=2800,learning_rate=0.01,max_depth=6,
                              min_child_weight=0,metric='mac',subsample=0.3,colsample_bytree=0.5 ) 

model_LGB.fit(data,y)

model_LGB.predict(data)



winsound.Beep(2500, 500)



######################################
# 測試(非正式)
#model_LGB = lgb.LGBMRegressor(n_estimators=2300,learning_rate=0.05,max_depth=10,min_child_weight=1,metric='mac' ) 
model_LGB = lgb.LGBMRegressor(n_estimators=2800,learning_rate=0.01,max_depth=6,
                              min_child_weight=0,metric='mac',subsample=0.3,colsample_bytree=0.5 ) 

# 檢查 overfitting
model_LGB.fit(train_x,train_y)
mean_absolute_error( model_LGB.predict(valid_x)   , valid_y  )
mean_absolute_error( model_LGB.predict(train_x)   , train_y  )




model_LGB.fit(data,y)

LGBM_train = model_LGB.predict(data)
LGBM_train[LGBM_train<0] = 0
mean_absolute_error(LGBM_train,y)


LGBM_pred = model_LGB.predict(pred)
LGBM_pred[LGBM_pred<0] = 0

submit_LGBM = output(  LGBM_pred )
submit_LGBM.to_csv("submit_LGBM_8_8.csv",index=False)








