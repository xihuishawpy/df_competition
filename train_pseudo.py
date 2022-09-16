import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import  lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
import warnings
from tqdm import tqdm
pd.set_option('display.max_columns', None)


# 计算列中0的占比
def col_zero_ratio(df):
    df_len = len(df)
    zero_stat = pd.DataFrame()
    stat_cols = [col for col in df.columns if col not in ['id', 'label']]
    zero_stat['stat_col'] = stat_cols
    zero_stat['zero_ratio'] = ((df[stat_cols] == 0).sum(axis=0) / df_len).values
    return zero_stat


def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, y)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(x_train,y_train)

        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)
        print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(y, oof_preds)))
    return test_preds / kfold



if __name__ == '__main__':
    train_df = pd.read_csv('data/dataTrain.csv')
    no_label_df = pd.read_csv('data/dataNoLabel.csv')
    test_df = pd.read_csv('data/dataA.csv')

    train_df['f3'] = train_df['f3'].map({'low': 1, 'mid': 2, 'high': 3})
    no_label_df['f3'] = no_label_df['f3'].map({'low': 1, 'mid': 2, 'high': 3})
    test_df['f3'] = test_df['f3'].map({'low': 1, 'mid': 2, 'high': 3})

    # # 计算列中0占比
    # train_df_zero = col_zero_ratio(train_df)
    # no_label_df_zero = col_zero_ratio(no_label_df)
    # test_df_zero = col_zero_ratio(test_df)
    #
    # data_zero_stat = pd.merge(train_df_zero, no_label_df_zero, how='left', on=['stat_col'], suffixes=('', '_no_label')) \
    #     .merge(test_df_zero, how='left', on=['stat_col'], suffixes=('', '_test'))
    # data_zero_stat['train-train_nolabel'] = data_zero_stat['zero_ratio'] - data_zero_stat['zero_ratio_no_label']
    # data_zero_stat['train-test'] = data_zero_stat['zero_ratio'] - data_zero_stat['zero_ratio_test']
    #
    # # 剔除0占比大于0.99的列
    # rm_cols = data_zero_stat[data_zero_stat['zero_ratio'] >= 0.5]['stat_col'].tolist()
    # fea_cols = [col for col in train_df.columns.values if col not in rm_cols + ['id', 'label']]
    #
    # train = train_df[fea_cols]
    # # 划分特征类型 ，类别数小于10为离散特征
    # cat_features = train.nunique()[train.nunique() < 10].index.tolist()
    # num_features = [col for col in fea_cols if col not in cat_features]
    # print(cat_features,num_features)

    # 暴力Feature 位置
    loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
    for df in [train_df, test_df,no_label_df]:
        for i in range(len(loc_f)):
            for j in range(i + 1, len(loc_f)):
                df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
                df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
                df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
                df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]] + 1)

    # 暴力Feature 通话
    com_f = ['f43', 'f44', 'f45', 'f46']
    for df in [train_df, test_df,no_label_df]:
        for i in range(len(com_f)):
            for j in range(i + 1, len(com_f)):
                df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
                df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
                df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
                df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]] + 1)

    feature_columns = [col for col in train_df.columns if col not in ['id','label']]
    train = train_df[feature_columns]
    y = train_df['label']
    test = test_df

    # pseudo label
    X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=2)
    # tuned_parameters = [{'n_estimators': range(100, 300, 500),
    #                      'max_depth': range(4, 8, 12),
    #                      'learning_rate': [0.01, 0.1]
    #                      }]

    # clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    predictions = clf.predict_proba(X_val)
    pre = predictions[:, 1]
    val_auc = roc_auc_score(y_val, pre)  # 验证集上的auc值
    print(val_auc)

    pseudo_preds = clf.predict(no_label_df[feature_columns])
    # pseudo_preds = pseudo_preds[:, 1]
    no_label_df['label'] = pseudo_preds
    train = train.append(no_label_df.iloc[:,1:-1]).reset_index()
    y = pd.Series(np.append(y,pseudo_preds))

    gbc = GradientBoostingClassifier()
    gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 60)
    print(gbc)

    # 剔除噪声数据 KFold=50~59，auc在0.5左右
    train = train[:50000]
    y = y[:50000]




    # 模型训练
    KF = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)
    feat_imp_df = pd.DataFrame({'feat': feature_columns, 'imp': 0})
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': 30,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 255,
        'verbose': -1,
        'seed': 2022,
        'bagging_seed': 2022,
        'feature_fraction_seed': 2022,
        'early_stopping_rounds': 100,

    }

    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros((len(test)))


    for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(train.iloc[trn_idx][feature_columns], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx][feature_columns], label=y.iloc[val_idx])
        num_round = 3000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,
        )

        oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][feature_columns], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(test[feature_columns], num_iteration=clf.best_iteration) / 5
        feat_imp_df['imp'] += clf.feature_importance() / 5

    print(f"AUC score: {roc_auc_score(y, oof_lgb)}")
    print(f"F1 score: {f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])}")
    print(f"Precision score: {precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])}")
    print(f"Recall score: {recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])}")

    # 提交结果
    test['label'] = predictions_lgb
    test[['id', 'label']].to_csv('sub2.csv', index=False)













