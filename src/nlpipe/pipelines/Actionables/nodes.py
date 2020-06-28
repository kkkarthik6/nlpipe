
'''Actionables'''
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier
import umap.umap_ as umap
import umap.plot as uplot
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
def gmm_train(x:np.ndarray,n_clusters):
    gmm=GaussianMixture(n_components=n_clusters)
    gmm.fit(x)
    return gmm
def RF_train(x:np.ndarray,y:list):
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(x,y)
    return clf
def gmm_predict(model,x):
    preds = model.predict(x)
    return preds
def model_predict(model,x):
    preds = model.predict(x)
    return preds

def umap_iplot(x, df_text,preds):
    hover_data = pd.DataFrame({'index': preds,
                               'label': df_text})
    mapper = umap.UMAP().fit(x)
    p = uplot.interactive(mapper, labels=preds, hover_data=hover_data, point_size=2)
    uplot.show(p)
    return mapper
def report_gen(y_test,preds):
    print(classification_report(y_test,preds))
    return classification_report(y_test,preds)
