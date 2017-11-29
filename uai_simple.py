import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb    

path='./'
result_path = './result/'

july_df=pd.read_csv(path+u'train_July.csv', sep=',', usecols=[4,9,10], dtype={4:np.int32, 9:str, 10:str})
aug_df=pd.read_csv(path+u'train_Aug.csv', sep=',', usecols=[4,9,10], dtype={4:np.int32, 9:str, 10:str})
train=pd.concat([july_df,aug_df], ignore_index=True, axis=0)
train = train[0:len(july_df)]
test=pd.read_csv(path+u'test_id_Aug_agg_public5k.csv')

cnt_dict = {}
for i in range(len(train)):
  item = (train.ix[i,'create_hour'], train.ix[i,'start_geo_id'], train.ix[i,'end_geo_id'])
  cnt_dict[item] = 1 if item not in cnt_dict else cnt_dict[item] + 1
test_cnt = []
for i in range(len(test)):
  item = (test.ix[i,'create_hour'], test.ix[i,'start_geo_id'], test.ix[i,'end_geo_id'])
  count = 1 if item not in cnt_dict else cnt_dict[item]/31
  test_cnt.append(count)
#print(len(test), len(test_cnt))
result = pd.DataFrame({'test_id': test['test_id'], 'count':test_cnt}, columns=['test_id', 'count'])
result.to_csv(result_path+'result.csv', index=False)
 


#mall_list=list(set(list(shop.mall_id)))
# result=pd.DataFrame()
#for mall in mall_list:
#    df_train=train1[train1.shop_id.notnull()]
#    df_test=train1[train1.shop_id.isnull()]
#    lbl = preprocessing.LabelEncoder()
#    lbl.fit(list(df_train['shop_id'].values))
#    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
#    num_class=df_train['label'].max()+1    
#    params = {
#            'objective': 'multi:softmax',
#            'eta': 0.1,
#            'max_depth': 9,
#            'eval_metric': 'merror',
#            'seed': 0,
#            'missing': -999,
#            'num_class':num_class,
#            'silent' : 1
#            }
#    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]    
#    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
#    xgbtest = xgb.DMatrix(df_test[feature])
#    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
#    num_rounds=60
#    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
#    df_test['label']=model.predict(xgbtest)
#    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
#    r=df_test[['row_id','shop_id']]
#    result=pd.concat([result,r])
#    result['row_id']=result['row_id'].astype('int')
#    result.to_csv(path+'sub.csv',index=False)
