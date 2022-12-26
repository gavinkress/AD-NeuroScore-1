# AD-NeuroScore Full Data Extraction Processing and Analysis Code
# Author: Gavin Kress, 11/09/2022
# Pacific Neuroscience Institute
# Inquiries to gkress@usc.edu

'''
LICENSE

AD_NeuroScore, Release 1.0 (c) 2022, The Pacific Neuroscience Institute and Foundation
(the "Software")

The Software remains the property of The Pacific Neuroscience Institute and Foundation
("the Foundation").

The Software is distributed "AS IS" under this License solely for non-commercial use in the hope that it will be useful, but in order that the Foundation as a charitable foundation protects its assets for the benefit of its educational and research purposes, the Foundation makes clear that no condition is made or to be implied, nor is any warranty given or to be implied, as to the accuracy of the Software, or that it will be suitable for any particular purpose or for use
under any specific conditions. Furthermore, the Foundation disclaims all responsibility for the use which is made of the Software. It further disclaims any liability for the outcomes arising from using the Software.

The Licensee agrees to indemnify the Foundation and hold the Foundation harmless from and against any and all claims, damages and liabilities asserted by third parties (including claims for negligence) which arise directly or indirectly from the use of the Software, or the sale of any products based on the Software.

No part of the Software may be reproduced, modified, transmitted, or transferred in any form or by any means, electronic or mechanical, without the express permission of the Foundation. The permission of the Foundation is not required if the said reproduction, modification, transmission, or transference is done without financial return, the conditions of this License are imposed upon the receiver of the product, and all original and amended source code is included in any transmitted product. You may be held legally responsible for any intellectual property infringement, including but not limited to copyright infringement, patent infringement and trademark infringement that is caused or encouraged by your failure to abide by these terms and conditions.

You are not permitted under this License to use this Software commercially. Use for which any financial return is received shall be defined as commercial use, and includes (1) integration of all or part of the source code or the Software into a product for sale or license by or on behalf of Licensee to third parties or (2) use of the Software or any derivative of it for research with the final aim of developing software products for sale or license to a third party or
(3) use of the Software or any derivative of it for research with the final aim of developing non-software products for sale or license to a third party, or (4) use of the Software to provide any service to an external organization for which payment is received. If you are interested in using the Software commercially, please contact the Foundation to negotiate a license. Contact details are: jbramen@pacificneuro.org quoting Reference Project AD-NeuroScore.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OUTDATED_IGNORE'] = '1'
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
import scipy
import similaritymeasures
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from itertools import groupby
from sklearn.naive_bayes import GaussianNB
import random
from sklearn import tree
from sklearn import svm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from IPython.display import display
import copy
from tqdm import tqdm
import pingouin as pg
import math
import statistics as st
import matplotlib.cbook
from matplotlib.lines import Line2D

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def tensorflow_quiet():

    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass


tensorflow_quiet()

############# PREPROCESSING #####################################
NR = 'NO'

# Load Data into Pandas Data Frames
proj_dir = r'C:\Users\gavin\PycharmProjects\ADNI'
filepath = r'C:\Users\gavin\OneDrive\Research\Pacific Neuroscience Institute\ADNI_Data_122221.xlsx'
filepath = r'C:\Users\gavin\OneDrive\Research\Pacific Neuroscience Institute\Long_Data_02.xlsx'
Scan_Info_DF = pd.read_excel(filepath, sheet_name='Sheet1')
Volume_DF_1 = pd.read_excel(filepath, sheet_name='QC cort_noQC subc ADNI3')
Volume_DF_2 = pd.read_excel(filepath, sheet_name='rough QC cort_subc ADNIGO 2')
Volume_DF = pd.concat([Volume_DF_1, Volume_DF_2], ignore_index=True)
Volume_DF_copy = pd.DataFrame(Volume_DF)

# Map to NeuroReader
NR_Map_DF = pd.read_excel(r'C:\Users\gavin\OneDrive\Research\Pacific Neuroscience Institute\NR_FS_ROIs.xlsx')
nonvol = ['SubjID', 'ICV']
NR_Vols = pd.DataFrame(columns=nonvol + NR_Map_DF['Neuroreader'].to_list(), index=Volume_DF.index)

for no in nonvol:
    NR_Vols[no] = Volume_DF[no]

for colll in NR_Map_DF['Neuroreader'].to_list():
    for ij in range(len(NR_Vols)):
        usm = 0
        lst = NR_Map_DF[NR_Map_DF['Neuroreader'] == colll]['Freesurfer Match(es)'].to_list()[0].split(", ")
        if len(lst)>1:
            for itlst in lst:
                usm = usm + Volume_DF.iloc[ij][itlst]
        else:
            usm = Volume_DF.iloc[ij][lst[0]]
        NR_Vols[colll][ij] = usm

NR_Vols = NR_Vols.drop(columns= ['mTIV'])

if NR == 'YES':
    proj_dir = r'C:\Users\gavin\PycharmProjects\ADNI\NR'
    Volume_DF = NR_Vols
    Volume_DF_copy = pd.DataFrame(Volume_DF)

# Remove surface areas and thicknesses from Data Frame
for col in Volume_DF_copy.columns:
    if col.__contains__('thick') or col.__contains__('surf') or col.__contains__('Thick') or col.__contains__('Surf'):
        Volume_DF = Volume_DF.drop(columns=col)

# Remove additions to the Subject ID labels
SubjectIDs = []
for subject in Volume_DF['SubjID']:
    SubjectIDs.append('_'.join(subject.split('_')[0:3]))
Volume_DF['SubjID'] = SubjectIDs

# Create DataFrame with covariates and DX
covariates = ['SEX', 'Age_full', 'Mfg Model']
Volume_DF = Volume_DF.join(Scan_Info_DF.set_index('SubjID')['DX'], on='SubjID')
for cov in covariates:
    Volume_DF = Volume_DF.join(Scan_Info_DF.set_index('SubjID')[cov], on='SubjID')
covariates.append('ICV')

# Reorder DF such that the first 5 columns are SubjID, DX, SEX, Age_full, Mfg Model, ICV
cols = Volume_DF.columns.tolist()
cols = [cols[0]] + cols[-4:] + cols[1:-4]
Volume_DF = Volume_DF[cols]

# Remove rows with missing or  not useful Data
Volume_DF = Volume_DF.dropna()
Volume_DF = Volume_DF[Volume_DF['SEX'] != 'X']
Volume_DF = Volume_DF[Volume_DF['DX'] != 'pending']
Volume_DF = Volume_DF.reset_index(drop=True)


# Data Harmonization
def convert_to_w(DF: pd.DataFrame, Covariates: list):  # Function which converts all volumes in DF to W-score
    # Create copy to store W-score
    DF_w = pd.DataFrame(DF)

    # Handle Categorical Variables
    # Create list of categorical covariates
    cat_cov = []
    num_cov = Covariates
    for cat_poss in ['SEX', 'Mfg Model', 'Manufacturer']:
        if cat_poss in Covariates:
            cat_cov.append(cat_poss)
            num_cov.remove(cat_poss)

    # One-Hot Encode Categorical Variables
    DF = pd.DataFrame(pd.get_dummies(DF, prefix=cat_cov, columns=cat_cov, drop_first=True))

    # Get list of numerical and categorical columns, combine
    cat_cols = []
    for cat in cat_cov:
        cat_cols = cat_cols + [icol for icol in DF.columns if cat in icol]
    final_covariates = num_cov + cat_cols

    # Variables that are not outcomes or covariates
    dif_vars = ['SubjID', 'DX']
    if 'ICV' in Covariates:
        pass
    else:
        dif_vars.append('ICV')

    # Obtain Volume Columns
    vol_cols = DF.columns.tolist()
    for re_col in (dif_vars+final_covariates):
        vol_cols.remove(re_col)
    reg_dict = {}
    for reg in vol_cols:  # for each region
        # Regress volume with covariates
        regr = linear_model.LinearRegression()
        regr.fit(DF[final_covariates], DF[reg])
        reg_dict[reg] = regr
        residuals = np.array(DF[reg])-regr.predict(DF[final_covariates])
        W = (residuals - np.mean(residuals))/np.std(residuals)
        DF_w[reg] = W
    import pickle
    with open('config.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(reg_dict, config_dictionary_file)
    return pd.DataFrame(DF_w)




normalize = False
if normalize:
    # Normalize Volumes
    Volume_DF.iloc[:, 6:] = Volume_DF.iloc[:, 6:].div(Volume_DF.ICV, axis=0)

    # Calculate W Statistic with regressors (sex, age, scanner type)
    Vol_W_DF = pd.DataFrame(convert_to_w(Volume_DF.copy(), covariates[:-1].copy()))
else:
    # Calculate W Statistic with regressors (sex, age, scanner type, ICV)
    Vol_W_DF = pd.DataFrame(convert_to_w(Volume_DF.copy(), covariates.copy()))

############# REGION IDENTIFICATION #####################################

# ANOVA to identify relevant brain regions
# Grabbing subset of patients
n = 50  # number of patients in each DX category used for ANOVA
p = 0.05/84  # p value threshold for significance // Bonferroni Correction to 0.05

# indexes of each diagnostic catagory
i_CN = Vol_W_DF[Vol_W_DF['DX']=='CN'].index.tolist()
i_Dem = Vol_W_DF[Vol_W_DF['DX']=='Dementia'].index.tolist()
i_MCI = Vol_W_DF[Vol_W_DF['DX']=='MCI'].index.tolist()

# Create lists of indices
anova_list = i_CN[0:n] + i_Dem[0:n] + i_MCI[0:n]
anova_list.sort()
var_list = i_CN[n:] + i_Dem[n:] + i_MCI[n:]
var_list.sort()

# Create DF for ANOVA and Analysis
Anova_DF = Vol_W_DF.iloc[anova_list, :].copy()
Vol_W_DF_Var = Vol_W_DF.iloc[var_list, :].copy()

# Create list of regions
reg_cols = Vol_W_DF.columns.tolist()
for re_col in (covariates+['SubjID', 'DX']):
   reg_cols.remove(re_col)

# Perform ANOVA to obtain list of ss regions
ss_regions = []
anova_z_table = []
for reg in reg_cols:
    result = scipy.stats.f_oneway(Anova_DF [reg][Anova_DF ['DX'] == 'CN'], Anova_DF [reg][Anova_DF ['DX'] == 'Dementia'], Anova_DF [reg][Anova_DF ['DX'] == 'MCI'])
    if result[1]<p:
        ss_regions.append(reg)
        anova_z_table.append({"Region": reg, "Z": -scipy.stats.norm.ppf(result[1]/2)})

############# COMPUTATION #####################################

# Get Average of 152 CN patients 76 Male and Female
CN_List = pd.DataFrame(Vol_W_DF_Var[Vol_W_DF_Var['DX']=='CN']['SubjID'])
M_List = pd.DataFrame(Scan_Info_DF[['SubjID', 'SEX']][Scan_Info_DF['SEX'] == 'M'])
F_List = pd.DataFrame(Scan_Info_DF[['SubjID', 'SEX']][Scan_Info_DF['SEX'] == 'F'])
M_CN_List = CN_List.set_index('SubjID').join(M_List.set_index('SubjID')).dropna().index.to_list()[0:76]
F_CN_List = CN_List.set_index('SubjID').join(F_List.set_index('SubjID')).dropna().index.to_list()[0:76]
Template_IDs = M_CN_List + F_CN_List

#Ensure age distribution compatability
Scan_Info_DF['Age_full'].mean()
Scan_Info_DF['Age_full'][Scan_Info_DF['SubjID'].isin(Template_IDs)].mean()

# Create Template and drop from Test Patients, get stats
Template = Vol_W_DF_Var[ss_regions][Vol_W_DF_Var['SubjID'].isin(Template_IDs)].mean()
Template_Demo = pd.DataFrame({'Mean': Vol_W_DF_Var[ss_regions][Vol_W_DF_Var['SubjID'].isin(Template_IDs)].mean(), "STD": Vol_W_DF_Var[ss_regions][Vol_W_DF_Var['SubjID'].isin(Template_IDs)].std()})
Template_Demo.to_excel(proj_dir + '\\' + 'Template Mean Std.xlsx')

Vol_W_DF_Var = Vol_W_DF_Var.drop(Vol_W_DF[Vol_W_DF['SubjID'].isin(Template_IDs)].index)

# Get p-value for each ss volume
ss_dict = {key: None for key in ss_regions}

Preg_DF = pd.DataFrame({'Dementia vs CN': ss_dict.copy(), 'Dementia vs MCI': ss_dict.copy(), 'MCI vs CN': ss_dict.copy()})

for reg in ss_regions:
    Preg_DF['MCI vs CN'][reg] = scipy.stats.ttest_ind(Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'CN'], Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'MCI']).pvalue
    Preg_DF['Dementia vs MCI'][reg] = scipy.stats.ttest_ind(Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'Dementia'], Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'MCI']).pvalue
    Preg_DF['Dementia vs CN'][reg] = scipy.stats.ttest_ind(Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'CN'], Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'Dementia']).pvalue


Preg_DF_z = pd.DataFrame(Preg_DF.mean(axis=1))
Preg_DF_z[0] = -scipy.stats.norm.ppf(Preg_DF_z[0])
Preg_DF_z = Preg_DF_z[0]

# Convert Regions p value to z-scores
Preg_DF_z2 = Preg_DF.copy()
for coll in Preg_DF.columns:
    for reg in ss_regions:
        Preg_DF_z2[coll][reg] = -scipy.stats.norm.ppf(Preg_DF[coll][reg]/2)


Sig_DF = pd.DataFrame({'CN': ss_dict.copy(),  'MCI': ss_dict.copy(), 'Dementia': ss_dict.copy(), 'Mean': ss_dict.copy()})
for reg in ss_regions:
        Sig_DF['CN'][reg] = Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'CN'].std()
        Sig_DF['MCI'][reg] = Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'MCI'].std()
        Sig_DF['Dementia'][reg] = Vol_W_DF_Var[reg][Vol_W_DF_Var['DX'] == 'Dementia'].std()
        Sig_DF['Mean'][reg] = np.array([Sig_DF['CN'][reg], Sig_DF['MCI'][reg], Sig_DF['Dementia'][reg]]).mean()

sig = Sig_DF['Mean']

Distance_DF = pd.DataFrame({'SubjID': [], 'DX': [], 'Euclidean': [], '1-D Hausdorff': [], 'N-D Hausdorff': [], '1-D Frechet': [], 'ADNeuro Score': [], 'N-D Frechet': []})


def fnn(x, tau, m, rtol, atol):  # False nearest neighbors algorithm
    L = len(x)
    Ra = np.std(x)
    FNN = np.zeros(m)
    for d in range(1, m + 1):
        M = L - d * tau - 1
        Y = np.zeros([M, d])

        for i in range(0, d):
            Y[:, i] = x[i * tau:(M + i * tau)]  # Create M vectors in d dimensions

        for n in range(1, M):

            y0 = np.ones([M, 1])

            distance = np.zeros([M, 2])
            distance[:, 0] = np.sqrt(
                np.sum(np.power(Y - y0.dot([Y[n, :]]), 2), axis=1))  # get distances of each vector from nth vector
            distance[:, 1] = range(M)
            ND = distance[np.argsort(distance[:, 0])]
            neardis = ND[:, 0]
            nearpos = ND[:, 1]
            nearpos = nearpos.astype(int)

            G = np.abs(x[n + (d) * tau] - x[nearpos[1] + (d) * tau])
            R = np.sqrt(G ** 2 + neardis[1] ** 2)

            if G / neardis[2] > rtol or R / Ra > atol:
                FNN[d - 1] = FNN[d - 1] + 1
    FNN = np.divide(FNN, FNN[0]) * 100
    return FNN


def ED(FNN): # function extracting embedding dimension
    for i in range(np.shape(FNN)[0]):
        EDD = FNN[i]
        if i > 0:
            if np.abs(FNN[i] - FNN[i - 1]) < 0.2:
                break
        else:
            EDD = FNN[i]
    EDD = i
    return EDD

def Embed(x, tau, d):  # function embedding series in d dimensions
    L = len(x)
    M = L - d * tau - 1
    Y = np.zeros([M, d])
    for i in range(0, d):
        Y[:, i] = x[i * tau:(M + i * tau)]
    return Y




domain =list(range(ss_regions.__len__()))

m = 10  # max dimensions
tau = int(0.05 * len(domain))  # time delay
rtol = 3  # threshold fold distance increase
atol = 2  # standard deviation threshold


for indx in Vol_W_DF_Var.index.to_list():
    new_pt = {'SubjID': [], 'DX': [], 'Euclidean': [], '1-D Hausdorff': [], 'N-D Hausdorff': [], '1-D Frechet': [], 'ADNeuro Score': [], 'N-D Frechet': []}

    volumes_W = Vol_W_DF_Var.loc[indx][ss_regions]
    new_pt['SubjID'] = Vol_W_DF_Var.loc[indx]['SubjID']
    new_pt['DX'] = Vol_W_DF_Var.loc[indx]['DX']
    new_pt['Euclidean'] = ((Template - volumes_W)**2).sum()

    #Creat pairs of points
    Temp2 = np.array([domain, Template.to_list()])
    Vols_2 = np.array([domain, volumes_W.to_list()])

    # Obtain distances from average
    new_pt['1-D Hausdorff'] = scipy.spatial.distance.directed_hausdorff(Temp2, Vols_2)[0]
    new_pt['1-D Frechet'] = similaritymeasures.frechet_dist(Vols_2, Temp2)

    #Embedding Dimension
    EDD = ED(fnn(Vols_2[1, :], tau, m, rtol, atol))
    VolsND = Embed(Vols_2[1, :], tau, EDD)
    TempND = Embed(Temp2[1, :], tau, EDD)
    new_pt['N-D Hausdorff'] = scipy.spatial.distance.directed_hausdorff(TempND, VolsND)[0]
    new_pt['N-D Frechet'] = similaritymeasures.frechet_dist(VolsND, TempND)
    new_pt['ADNeuro Score'] = math.sqrt((((Template - volumes_W)**2)*(Preg_DF_z)**2).sum())
    new_pt['Std adj ADNeuro Score'] = math.sqrt((((Template - volumes_W) ** 2) * (Preg_DF_z/(10*sig**2)) ** 2).sum())


    #Adding Pt to data structure
    Distance_DF = Distance_DF.append(new_pt, ignore_index=True)

# Saving weights
pd.DataFrame(Preg_DF_z/(10*sig**2)).to_excel(proj_dir + '\\' + 'Adj_Z_Weights_Table.xlsx')

# Obtaining Education Adjusted MMSE

MMSE_DF = Scan_Info_DF[['SubjID', 'PTEDUCAT', 'MMSE', 'DX']].dropna().copy().reset_index(drop = True)
MMSE_DF['MMSE_Cat'] = ""

for pt in range(MMSE_DF.__len__()):
    YOE = int(np.array(MMSE_DF['PTEDUCAT'][pt]))
    MMSE = float(np.array(MMSE_DF['MMSE'][pt]))
    cut_off = [23, 18, 9]
    if YOE<7:
        cut_off[0] = 22
    elif YOE>=7 and YOE<12:
        cut_off[0] = 24
    elif YOE==12:
        cut_off[0] = 25
    elif YOE>12:
        cut_off[0] = 26

    if MMSE > cut_off[0]:
        MMSE_DF['MMSE_Cat'][pt] = 0
    elif MMSE > cut_off[1] and MMSE <= cut_off[0]:
        MMSE_DF['MMSE_Cat'][pt] = 1
    elif MMSE > cut_off[2] and MMSE <= cut_off[1]:
        MMSE_DF['MMSE_Cat'][pt] = 2
    elif MMSE <= cut_off[2]:
        MMSE_DF['MMSE_Cat'][pt] = 3
MMSE_DF['MMSE_Cat'].value_counts()

# Metrics ################################################
#'NPI_Score', 'CDR_Global', 'ADNI_EF', 'GDS_total', 'MMSE', 'ADNI_MEM', 'ADAS11'
Metrics = ['CDRSB', 'MMSE', 'ADAS11']

# Adding other metrics to distance DF

if NR == 'YES':
    AHV_DF = pd.DataFrame({'SubjID': Volume_DF['SubjID'].to_list(), 'AHV': ((Volume_DF['Right Hippocampus'] + Volume_DF['Left Hippocampus'])/ Volume_DF['ICV']).to_list()})
else:
    AHV_DF = pd.DataFrame({'SubjID': Volume_DF['SubjID'].to_list(),'AHV': ((Volume_DF['Rhippo'] + Volume_DF['Lhippo']) / Volume_DF['ICV']).to_list()})
New_Distance_DF = pd.merge(left=Distance_DF, right=Scan_Info_DF[['SubjID'] + Metrics], left_on='SubjID', right_on='SubjID')
New_Distance_DF = pd.merge(left=New_Distance_DF.copy(), right=AHV_DF, left_on='SubjID', right_on='SubjID')
New_Distance_DF = New_Distance_DF.dropna(subset=Metrics)
if 'ADAS11' in Metrics:
    try:
        New_Distance_DF = New_Distance_DF[New_Distance_DF.ADAS11.str.contains("NA").isna()]
    except:
        pass
Distance_DF = New_Distance_DF.copy()
Distance_DF = Distance_DF.reset_index(drop = True)


# Encode DX
encoder = preprocessing.LabelEncoder()
encoder.fit(Distance_DF['DX'])
encoded_Y = encoder.transform(Distance_DF['DX'])


# Get Demographics

Experimental_Demo = pd.merge(left=Distance_DF, right=Scan_Info_DF[['SubjID', 'SEX', 'PTEDUCAT', 'Age_full']], left_on='SubjID', right_on='SubjID')
Experimental_Demo2 = pd.merge(left=Distance_DF, right=Scan_Info_DF[['SubjID', 'DX_m12', 'DX_m24', 'SEX', 'PTEDUCAT', 'Age_full']], left_on='SubjID', right_on='SubjID')
Experimental_Demo3 = Experimental_Demo2.drop(['DX', 'DX_m12'], axis=1).rename(columns={'DX_m24': 'DX'}).dropna(subset = ['DX'])
Experimental_Demo2 = Experimental_Demo2.drop(['DX', 'DX_m24'], axis=1).rename(columns={'DX_m12': 'DX'}).dropna(subset = ['DX'])
Experimental_Demo2 = Experimental_Demo2[Experimental_Demo2['DX'] != 0]
Experimental_Demo3 = Experimental_Demo3[Experimental_Demo3['DX'] != 0]
Anova_Demo = pd.merge(left=Anova_DF, right=Scan_Info_DF[['SubjID', 'PTEDUCAT']], left_on='SubjID', right_on='SubjID')
Template_Demo = Scan_Info_DF[Scan_Info_DF['SubjID'].isin(Template_IDs)]

for ct, DF in enumerate([Experimental_Demo, Anova_Demo, Template_Demo, Experimental_Demo2, Experimental_Demo3]):
    DX_Cat = {'CN':'', 'MCI': '', 'Dementia': '', 'Total': ''}
    Demo_Table = {'N': DX_Cat.copy(), 'Age': DX_Cat.copy(), '% Male': DX_Cat.copy(), 'YOE': DX_Cat.copy()}
    for DX in list(DX_Cat.keys()):
        if DX == 'Total':
            Demo_Table['N'][DX] = DF.__len__()
            Demo_Table['Age'][DX] = str(DF['Age_full'].mean()) + " (" + str(DF['Age_full'].std()) + ")" + " (" + str(DF['Age_full'].min()) + "-" + str(DF['Age_full'].max())+ ")"
            Demo_Table['YOE'][DX] = str(DF['PTEDUCAT'].mean()) + " (" + str(DF['PTEDUCAT'].std()) + ")"
            Demo_Table['% Male'][DX] = DF['SEX'].value_counts()['M']/DF.__len__()
        else:
            if ct == 2:
                pass
            else:
                Demo_Table['N'][DX] = DF['DX'].value_counts()[DX]
                Demo_Table['Age'][DX] = str(DF['Age_full'][DF['DX'] ==DX].mean())+ " (" + str(DF['Age_full'][DF['DX'] ==DX].std()) + ")"  + " (" + str(DF['Age_full'][DF['DX'] ==DX].min()) + "-" + str(DF['Age_full'][DF['DX'] ==DX].max())+ ")"
                Demo_Table['YOE'][DX] = str(DF['PTEDUCAT'][DF['DX'] == DX].mean())+ " (" + str(DF['PTEDUCAT'][DF['DX'] == DX].std()) + ")"
                Demo_Table['% Male'][DX] = DF['SEX'][DF['DX'] ==DX].value_counts()['M']/DF['DX'].value_counts()[DX]
    if ct == 0:
        var = 'Experimental_Demo_Baseline'
    elif ct == 1:
        var = 'Anova_Demo'
    elif ct == 2:
        var = 'Template_Demo'
    elif ct == 3:
        var = 'Experimental_Demo_12_Months'
    elif ct == 4:
        var = 'Experimental_Demo_24_Months'
    pd.DataFrame(Demo_Table).to_excel(proj_dir + '\\' + var + '_Table.xlsx')



############# ANALYSIS AND RESULTS #####################################
# Lists
distances = ['Euclidean', '1-D Frechet', '1-D Hausdorff', 'N-D Frechet', 'N-D Hausdorff', 'ADNeuro Score', 'AHV']
distances_dict = {'Euclidean': '', '1-D Hausdorff': '', 'N-D Hausdorff': '', '1-D Frechet': '', 'N-D Frechet': '', 'ADNeuro Score': '', 'AHV': ''}

#Violin Plot

fig, ax = plt.subplots(len(distances), 1, figsize = (6, distances.__len__()*4))
for c, d in enumerate(distances):
    sns.violinplot(ax = ax[c], x = 'DX', y = d, data = Distance_DF[['DX']+distances], order=['CN', 'MCI', 'Dementia'])
plt.show()

# Creating Table of R^2 coefficients between distance metrics and cognitive scores

R2_DF = {}
for M in Metrics:
    R2_DF[M] = distances_dict.copy()

R2_DF_CN = copy.deepcopy(R2_DF)
R2_DF_MCI = copy.deepcopy(R2_DF)
R2_DF_Dementia = copy.deepcopy(R2_DF)

fig, ax = plt.subplots(Metrics.__len__(), distances.__len__(), figsize = (distances.__len__()*5, Metrics.__len__()*5))
for i, met in enumerate(Metrics):
    for j, dis in enumerate(distances):
        x = np.array(Distance_DF[dis].to_list())
        y = np.array(Distance_DF[met].to_list())
        res = scipy.stats.linregress(x, y)

        xc = np.array(Distance_DF[Distance_DF['DX']=='CN'][dis].to_list())
        yc = np.array(Distance_DF[Distance_DF['DX']=='CN'][met].to_list())
        resc = scipy.stats.linregress(xc, yc)

        xm = np.array(Distance_DF[Distance_DF['DX']=='MCI'][dis].to_list())
        ym = np.array(Distance_DF[Distance_DF['DX']=='MCI'][met].to_list())
        resm = scipy.stats.linregress(xm, ym)

        xd = np.array(Distance_DF[Distance_DF['DX']=='Dementia'][dis].to_list())
        yd = np.array(Distance_DF[Distance_DF['DX']=='Dementia'][met].to_list())
        resd = scipy.stats.linregress(xd, yd)

        R2_DF[met][dis] = str(res.rvalue) +  " (p = " + str(res.pvalue) + ")" + " n = " + str(len(x))
        R2_DF_CN[met][dis] = str(resc.rvalue) + " (p = " + str(resc.pvalue) + ")" + " n = " + str(len(xc))
        R2_DF_MCI[met][dis] = str(resm.rvalue) + " (p = " + str(resm.pvalue) + ")" + " n = " + str(len(xm))
        R2_DF_Dementia[met][dis] = str(resd.rvalue) + " (p = " + str(resd.pvalue) + ")" + " n = " + str(len(xd))
        ax[i,j].plot(x, y, 'r.', label='Original data')
        ax[i,j].plot(x, res.intercept + res.slope * x, 'k', label='Fitted line')
        ax[i,j].set_ylabel(met.replace('_', " "))
        ax[i,j].set_xlabel(dis)
        ax[i,j].legend()
plt.show()

# Creating Table of P values and Log reg AUC for each comparison and distance metric
distance_list = ['Euclidean', '1-D Hausdorff', 'N-D Hausdorff', '1-D Frechet', 'N-D Frechet'] + Metrics + ['ADNeuro Score', 'AHV']
distance_dict = {}
distance_dict_L = {}
for d in distance_list:
    distance_dict[d] = ''
    distance_dict_L[d] = []

distance_list2 = ['Euclidean', '1-D Hausdorff', 'N-D Hausdorff', '1-D Frechet', 'N-D Frechet','ADNeuro Score', 'AHV']
pairwise_dict = {'CN vs Dementia': distance_dict.copy(), 'MCI vs Dementia': distance_dict.copy(), 'CN vs MCI': distance_dict.copy()}

log_AUC_dict = pd.DataFrame(pairwise_dict.copy())
Youden_AUC_dict = pd.DataFrame(pairwise_dict.copy())
Cutoff_AUC_dict = pd.DataFrame(pairwise_dict.copy())
def tt_split(x, y, train_size):
    if np.shape(x) == np.shape(y):
        uniques = np.unique(y, return_counts=True)
        tr = []
        for ct, u in enumerate(uniques[0]):
            inds = list(np.argwhere(y == u).flatten())
            tr = tr+ random.sample(inds, int(train_size * float(len(inds))))

        tr = np.array(tr).flatten()
        te = [aa for aa in range(0, len(x)) if aa not in tr]
        return x[tr], x[te], y[tr], y[te]
    else:
        print('shape not the same')
        return -1

def Youden_index_analysis(tpr_i, fpr_i): #returns youden index, index of optimal threshold and fpr, tpr at that index, as well as auc up to that index
    w = 0
    a = tpr_i + 1-fpr_i
    inde = np.where(a==a.max())[0][0]
    while tpr_i[inde] == 0 or fpr_i[inde] == 1:
        w = w - 1
        s = a.copy()
        s.sort()
        inde = np.where(a==s[w])[0][0]
    if inde != 0:
        x = fpr_i[:inde].copy()
        y = tpr_i[:inde].copy()
        places = []
        for t in x:
            index = max([i for i, xr in enumerate(x) if xr == t])
            if index in places:
                pass
            else:
                places.append(index)

        roc = scipy.integrate.simpson(y[places], x[places])
        norm_roc = roc/fpr_i[inde]
        return inde, {'Youden index': a[inde], 'FPR': fpr_i[inde], 'TPR': tpr_i[inde], 'Norm_AUC': norm_roc}
    else:
        return inde, {'Youden index': a[inde], 'FPR': fpr_i[inde], 'TPR': tpr_i[inde], 'Norm_AUC': 1}



def Cutoff_index_analysis(tpr_i, fpr_i): #returns cutoff index, index of optimal threshold and fpr, tpr at that index, as well as auc up to that index
    w = 0
    acc = 0.15
    a = (fpr_i - acc)**2
    inde = np.where(a==a.min())[0][0]
    while tpr_i[inde] == 0 or fpr_i[inde] == 1:
        w = w - 1
        s = a.copy()
        s.sort()
        inde = np.where(a==s[w])[0][0]
    if inde != 0:
        x = fpr_i[:inde].copy()
        y = tpr_i[:inde].copy()
        places = []
        for t in x:
            index = max([i for i, xr in enumerate(x) if xr == t])
            if index in places:
                pass
            else:
                places.append(index)

        roc = scipy.integrate.simpson(y[places], x[places])
        norm_roc = roc/fpr_i[inde]
        return inde, {'Cutoff index': inde, 'FPR': fpr_i[inde], 'TPR': tpr_i[inde], 'Norm_AUC': norm_roc}
    else:
        return inde, {'Cutoff index': inde, 'FPR': fpr_i[inde], 'TPR': tpr_i[inde], 'Norm_AUC': 1}

plt.figure(num=1, figsize=(len(distance_list)*3, 9))
plt.figure(num=2, figsize=(4, 12), dpi = 500)
a = 0.2
inti = 0.85

Array1 = np.array(['ADNeuro Score', 'AHV'])
Array2 = np.array(['TPR','FPR'])
ind_array = [Array1.repeat(len(Array2)), np.tile(Array2, len(Array1))]
AUC_Comp_DF = pd.DataFrame(columns=['CN vs Dementia', 'CN vs MCI', 'MCI vs Dementia'], index=ind_array)

for cc, distance in enumerate(distance_list):
    cn = Distance_DF[distance][Distance_DF['DX']=='CN']
    Dem = Distance_DF[distance][Distance_DF['DX']=='Dementia']
    mci = Distance_DF[distance][Distance_DF['DX'] == 'MCI']

    ncn = cn.__len__()
    nmci = mci.__len__()
    nDem = Dem.__len__()

    d1 = pg.compute_effsize(cn.to_list(), Dem.to_list(), eftype='cohen')
    d2 = pg.compute_effsize(mci.to_list(), Dem.to_list(), eftype='cohen')
    d3 = pg.compute_effsize(cn.to_list(), mci.to_list(), eftype='cohen')

    v1 = (((ncn +nmci)/(ncn*nmci) + d1**2*(2*(ncn+nmci-2))**-1)*((ncn +nmci)/(ncn+nmci-2)))**0.5
    v2 = (nDem +nmci)/(nDem*nmci) + d1**2*(2*(nDem+nmci-2))**-1*((nDem +nmci)/(nDem+nmci-2))
    v3 = (ncn +nDem)/(ncn*nDem) + d1**2*(2*(ncn+nDem-2))**-1*((ncn +nDem)/(ncn+nDem-2))

    r1 = str(d1 - 1.96*v1) + "-" + str(d1 + 1.96*v1)
    r2 = str(d2 - 1.96 * v2) + "-" + str(d2 + 1.96 * v2)
    r3 = str(d3 - 1.96 * v3) + "-" + str(d3 + 1.96 * v3)

    pairwise_dict['CN vs Dementia'][distance] = str(scipy.stats.ttest_ind(cn, Dem)[1]) + " (" + str(scipy.stats.norm.ppf(scipy.stats.ttest_ind(cn, Dem)[1]/2)) + ")"  + " d = " + str(d1) + " (" + r1 + ")"
    pairwise_dict['MCI vs Dementia'][distance] = str(scipy.stats.ttest_ind(mci , Dem)[1]) + " (" + str(scipy.stats.norm.ppf(scipy.stats.ttest_ind(mci, Dem)[1]/2)) + ")" + " d = " + str(d2) + " (" + r2 + ")"
    pairwise_dict['CN vs MCI'][distance] = str(scipy.stats.ttest_ind(cn, mci)[1]) + " (" + str(scipy.stats.norm.ppf(scipy.stats.ttest_ind(cn, mci)[1]/2)) + ")" + " d = " + str(d3) + " (" + r3 + ")"


    # Log Reg
    rs_n = 2
    ts = 0.8
    kf = 50

    #CN vs Dementia
    lr_auc_list = []
    fp_list1 = []
    tp_list1 = []
    plt.figure(1)
    plt.subplot(3,len(distance_list),cc+1)
    for i in range(kf):
        X = pd.concat([Distance_DF[[distance, 'DX']][Distance_DF['DX']=='CN'], Distance_DF[[distance, 'DX' ]][Distance_DF['DX']=='Dementia']]).copy()
        y = np.array(X['DX'].copy())
        X = np.array(X[distance].copy())
        trainX, testX, trainy, testy = tt_split(X, y, train_size=ts)
        model = LogisticRegression(solver='lbfgs')
        model.fit(np.array(trainX).reshape(-1,1), np.array(trainy))
        lr_probs = model.predict_proba(np.array(testX).reshape(-1,1))
        lr_probs = lr_probs[:, 1]
        lr_auc = roc_auc_score(testy, lr_probs)
        fpr, tpr, thresholds = roc_curve(testy, lr_probs, pos_label='Dementia', drop_intermediate=False)
        lr_auc_list.append(lr_auc)
        plt.figure(1)
        plt.plot(fpr, tpr,  color='0.5')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(distance + '\nCN vs Dementia')

        if distance == 'ADNeuro Score':
            plt.figure(2)
            plt.subplot(3, 1, 1)
            plt.plot(fpr, tpr, color=(inti, 0, 0, a))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('CN vs Dementia')
        if distance == 'AHV':
            plt.figure(2)
            plt.subplot(3, 1, 1)
            plt.plot(fpr, tpr, color=(0, 0, inti, a))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('CN vs Dementia')



        fp_list1.append(fpr)
        tp_list1.append(tpr)

    meann = np.array(lr_auc_list).mean()
    std = np.array(lr_auc_list).std()

    tp_list1 = [x for x in tp_list1 if len(x) == st.mode([len(x) for x in tp_list1])]
    fp_list1 = [x for x in fp_list1 if len(x) == st.mode([len(x) for x in fp_list1])]
    tp_m = np.stack(tp_list1).mean(axis = 0)
    fp_m = np.stack(fp_list1).mean(axis=0)
    plt.figure(1)
    plt.subplot(3, len(distance_list), cc + 1)
    plt.plot(fp_m, tp_m, color='r')

    if distance == 'ADNeuro Score':
        plt.figure(2)
        plt.subplot(3, 1, 1)
        plt.plot(fp_m, tp_m, color='r', lw = 2, zorder = 1502)
        plt.plot(fp_m, tp_m, color='k', lw = 5, zorder=1501)
        AUC_Comp_DF['CN vs Dementia']['ADNeuro Score']['TPR'] = tp_m
        AUC_Comp_DF['CN vs Dementia']['ADNeuro Score']['FPR'] = fp_m
    if distance == 'AHV':
        plt.figure(2)
        plt.subplot(3, 1, 1)
        plt.plot(fp_m, tp_m, color='b', lw = 2, zorder = 1500)
        plt.plot(fp_m, tp_m, color='k', lw = 5, zorder=1499)
        AUC_Comp_DF['CN vs Dementia']['AHV']['TPR'] = tp_m
        AUC_Comp_DF['CN vs Dementia']['AHV']['FPR'] = fp_m

    #meann1 = np.array(fp_list).mean()
    #std1 = np.array(fp_list).std()

    #meann2 = np.array(tp_list).mean()
    #std2 = np.array(tp_list).std()

    log_AUC_dict['CN vs Dementia'][distance] = str(meann) + " (" + str(meann - 1.96*std) + "-" + str(meann + 1.96*std) +")"
    indss, dddi = Youden_index_analysis(tp_m, fp_m)
    Youden_AUC_dict['CN vs Dementia'][distance] = dddi
    indsss, dddii = Cutoff_index_analysis(tp_m, fp_m)
    Cutoff_AUC_dict['CN vs Dementia'][distance] = dddii
    #                                         + " FPR "
    #str(meann1) + " (" + str(meann1 - 1.96 * std1) + "-" + str(meann1 + 1.96 * std1) + ")" + " TPR "
    #str(meann2) + " (" + str(meann2 - 1.96 * std2) + "-" + str(meann2 + 1.96 * std2) + ")"


    #MCI vs Dementia
    lr_auc_list = []
    fp_list2 = []
    tp_list2 = []
    plt.figure(1)
    plt.subplot(3,len(distance_list),len(distance_list)+ cc+1)
    for i in range(kf):
        X = pd.concat([Distance_DF[[distance, 'DX']][Distance_DF['DX']=='MCI'], Distance_DF[[distance, 'DX' ]][Distance_DF['DX']=='Dementia']]).copy()
        y = np.array(X['DX'].copy())
        X = np.array(X[distance].copy())
        trainX, testX, trainy, testy = tt_split(X, y, train_size=ts)
        model = LogisticRegression(solver='lbfgs')
        model.fit(np.array(trainX).reshape(-1,1), np.array(trainy))
        lr_probs = model.predict_proba(np.array(testX).reshape(-1,1))
        lr_probs = lr_probs[:, 1]
        lr_auc = roc_auc_score(testy, lr_probs)
        fpr, tpr, thresholds = roc_curve(testy, lr_probs, pos_label='MCI', drop_intermediate=False)
        lr_auc_list.append(lr_auc)
        fp_list2.append(fpr)
        tp_list2.append(tpr)
        plt.figure(1)
        plt.plot(fpr, tpr,  color='0.5')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MCI vs Dementia')

        if distance == 'ADNeuro Score':
            plt.figure(2)
            plt.subplot(3, 1, 2)
            plt.plot(fpr, tpr, color=(inti, 0, 0, a))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('MCI vs Dementia')
        if distance == 'AHV':
            plt.figure(2)
            plt.subplot(3, 1, 2)
            plt.plot(fpr, tpr, color=(0, 0, inti, a))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('MCI vs Dementia')


    meann = np.array(lr_auc_list).mean()
    std = np.array(lr_auc_list).std()
    tp_list2 = [x for x in tp_list2 if len(x) == st.mode([len(x) for x in tp_list2])]
    fp_list2 = [x for x in fp_list2 if len(x) == st.mode([len(x) for x in fp_list2])]
    tp_m = np.stack(tp_list2).mean(axis = 0)
    fp_m = np.stack(fp_list2).mean(axis=0)
    plt.figure(1)
    plt.subplot(3, len(distance_list), len(distance_list) + cc + 1)
    plt.plot(fp_m, tp_m, color='r')

    if distance == 'ADNeuro Score':
        plt.figure(2)
        plt.subplot(3, 1, 2)
        plt.plot(fp_m, tp_m, color='r', lw = 2, zorder = 1502)
        plt.plot(fp_m, tp_m, color='k', lw = 5, zorder=1501)
        AUC_Comp_DF['MCI vs Dementia']['ADNeuro Score']['TPR'] = tp_m
        AUC_Comp_DF['MCI vs Dementia']['ADNeuro Score']['FPR'] = fp_m
    if distance == 'AHV':
        plt.figure(2)
        plt.subplot(3, 1, 2)
        plt.plot(fp_m, tp_m, color='b', lw = 2, zorder = 1500)
        plt.plot(fp_m, tp_m, color='k', lw = 5, zorder=1499)
        AUC_Comp_DF['MCI vs Dementia']['AHV']['TPR'] = tp_m
        AUC_Comp_DF['MCI vs Dementia']['AHV']['FPR'] = fp_m

    log_AUC_dict['MCI vs Dementia'][distance] = str(meann) + " (" + str(meann - 1.96*std) + "-" + str(meann + 1.96*std) +")"
    indss, dddi = Youden_index_analysis(tp_m, fp_m)
    Youden_AUC_dict['MCI vs Dementia'][distance] = dddi
    indsss, dddii = Cutoff_index_analysis(tp_m, fp_m)
    Cutoff_AUC_dict['MCI vs Dementia'][distance] = dddii
    #MCI vs CN
    lr_auc_list = []
    fp_list3 = []
    tp_list3 = []
    plt.figure(1)
    plt.subplot(3,len(distance_list),2*len(distance_list)+ cc+1)
    for i in range(kf):
        X = pd.concat([Distance_DF[[distance, 'DX']][Distance_DF['DX']=='MCI'], Distance_DF[[distance, 'DX' ]][Distance_DF['DX']=='CN']]).copy()
        y = np.array(X['DX'].copy())
        X = np.array(X[distance].copy())
        trainX, testX, trainy, testy = tt_split(X, y, train_size=ts)
        model = LogisticRegression(solver='lbfgs')
        model.fit(np.array(trainX).reshape(-1,1), np.array(trainy))
        lr_probs = model.predict_proba(np.array(testX).reshape(-1,1))
        lr_probs = lr_probs[:, 1]
        lr_auc = roc_auc_score(testy, lr_probs)
        fpr, tpr, thresholds = roc_curve(testy, lr_probs, pos_label='MCI', drop_intermediate=False)
        lr_auc_list.append(lr_auc)
        fp_list3.append(fpr)
        tp_list3.append(tpr)
        plt.figure(1)
        plt.plot(fpr, tpr,  color='0.5')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('CN vs MCI')

        if distance == 'ADNeuro Score':
            plt.figure(2)
            plt.subplot(3, 1, 3)
            plt.plot(fpr, tpr, color=(inti, 0, 0, a))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('CN vs MCI')
        if distance == 'AHV':
            plt.figure(2)
            plt.subplot(3, 1, 3)
            plt.plot(fpr, tpr, color=(0, 0, inti, a))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('CN vs MCI')

    meann = np.array(lr_auc_list).mean()
    std = np.array(lr_auc_list).std()
    tp_list3 = [x for x in tp_list3 if len(x) == st.mode([len(x) for x in tp_list3])]
    fp_list3 = [x for x in fp_list3 if len(x) == st.mode([len(x) for x in fp_list3])]
    tp_m = np.stack(tp_list3).mean(axis = 0)
    fp_m = np.stack(fp_list3).mean(axis=0)
    plt.figure(1)
    plt.subplot(3, len(distance_list), 2 * len(distance_list) + cc + 1)
    plt.plot(fp_m, tp_m, color='r')
    if distance == 'ADNeuro Score':
        plt.figure(2)
        plt.subplot(3, 1, 3)
        plt.plot(fp_m, tp_m, color='r', lw = 2, zorder = 1502)
        plt.plot(fp_m, tp_m, color='k', lw = 5, zorder=1501)
        AUC_Comp_DF['CN vs MCI']['ADNeuro Score']['TPR'] = tp_m
        AUC_Comp_DF['CN vs MCI']['ADNeuro Score']['FPR'] = fp_m
    if distance == 'AHV':
        plt.figure(2)
        plt.subplot(3, 1, 3)
        plt.plot(fp_m, tp_m, color='b', lw = 2, zorder = 1500)
        plt.plot(fp_m, tp_m, color='k', lw = 5, zorder=1499)
        AUC_Comp_DF['CN vs MCI']['AHV']['TPR'] = tp_m
        AUC_Comp_DF['CN vs MCI']['AHV']['FPR'] = fp_m

    log_AUC_dict['CN vs MCI'][distance] = str(meann) + " (" + str(meann - 1.96*std) + "-" + str(meann + 1.96*std) +")"
    indss, dddi = Youden_index_analysis(tp_m, fp_m)
    Youden_AUC_dict['CN vs MCI'][distance] = dddi
    indsss, dddii = Cutoff_index_analysis(tp_m, fp_m)
    Cutoff_AUC_dict['CN vs MCI'][distance] = dddii

for i in range(3):
    plt.figure(2)
    plt.subplot(3, 1, i+1)
    legend_elements = [Line2D([0], [0], color='r', label="ADNeuro Score"), Line2D([0], [0], color='b', label="AHV")]
    plt.legend(handles=legend_elements, loc= 'lower right')

Accuracy_DF = {'Gaussian Naive Bayes': distance_dict_L.copy(), 'Decision Tree': distance_dict_L.copy(), 'Support Vector Machine': distance_dict_L.copy(), 'Neural Network': distance_dict_L.copy()}
Accuracy_DF['Gaussian Naive Bayes']['All DistancesG'] = []
Accuracy_DF['Decision Tree']['All DistancesD'] = []
Accuracy_DF['Support Vector Machine']['All DistancesS'] = []
Accuracy_DF['Neural Network']['All DistancesN'] = []

plt.show(num=1)
plt.show(num=2)


#Saving Youden index result
Array1 = np.array(Youden_AUC_dict.index.to_list())
Array2 = np.array(list(Youden_AUC_dict[Youden_AUC_dict.columns.to_list()[0]][Array1[0]].keys()))
ind_array = [Array1.repeat(len(Array2)), np.tile(Array2, len(Array1))]
Youden_AUC_DF = pd.DataFrame(columns=Youden_AUC_dict.columns.to_list(), index=ind_array)


for i in Youden_AUC_DF.columns:
    for j in Array1:
        for k in Array2:
            Youden_AUC_DF.loc[(j, k), i] = Youden_AUC_dict[i][j][k]


Youden_AUC_DF.to_excel(proj_dir + '\\' + 'Youden_AUC_Table.xlsx')


#Saving Cutoff index result
Array1 = np.array(Cutoff_AUC_dict.index.to_list())
Array2 = np.array(list(Cutoff_AUC_dict[Cutoff_AUC_dict.columns.to_list()[0]][Array1[0]].keys()))
ind_array = [Array1.repeat(len(Array2)), np.tile(Array2, len(Array1))]
Cutoff_AUC_DF = pd.DataFrame(columns=Cutoff_AUC_dict.columns.to_list(), index=ind_array)


for i in Cutoff_AUC_DF.columns:
    for j in Array1:
        for k in Array2:
            Cutoff_AUC_DF.loc[(j, k), i] = Cutoff_AUC_dict[i][j][k]


Cutoff_AUC_DF.to_excel(proj_dir + '\\' + 'Cutoff_AUC_Table.xlsx')

k = 10  # fold testing
def create_model():
    model = Sequential()  # create model
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu', input_dim=np.shape(X[Train])[1]))
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu'))
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu'))
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model
    return model

def create_model2():
    model = Sequential()  # create model
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu', input_dim=np.shape(X[Train])[1]))
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu'))
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu'))
    model.add(Dense(round(np.shape(X[Train])[1]), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # Compile model
    return model

def Compare_Corr(r1, r2, n1, n2):
    r1 = abs(r1)
    r2 = abs(r2)
    if n1>3 and n2>3:
        z1 = 0.5*(np.log(1+r1)-np.log(1-r1))
        z2 = 0.5*(np.log(1+r2)-np.log(1-r2))
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z = np.abs(z1 - z2)/se
        p = (1-scipy.stats.norm.cdf(z))
        return p
    else:
        return "N/A"


for hi in range(k):
    # Obtain Training indices
    NT = [150, 200, 60]  # number of training patients from each group
    Train = random.sample(Distance_DF[Distance_DF['DX'] == 'CN'].index.tolist(), NT[0]) + random.sample(Distance_DF[Distance_DF['DX'] == 'MCI'].index.tolist(), NT[1]) + random.sample(Distance_DF[Distance_DF['DX'] == 'Dementia'].index.tolist(), NT[2])
    Test = [x for x in range(0, Distance_DF.__len__()) if x not in Train]

    # Creating Table of ML Algorithm accuracy values for each comparison and distance metric
    dict_ML = {'Gaussian Naive Bayes': distance_dict.copy(), 'Decision Tree': distance_dict.copy(), 'Support Vector Machine': distance_dict.copy(), 'Neural Network': distance_dict.copy()}
    for dist in distance_list:
        X = np.array(Distance_DF[dist]).reshape(-1, 1)

        # GNB
        gnb = GaussianNB()
        GNB = gnb.fit(X[Train], encoded_Y[Train])
        Predicted = GNB.predict(X[Test])
        Actual = np.array(encoded_Y[Test])
        Accuracy = np.sum((Predicted == Actual) * 1.) / len(Predicted)
        Accuracy_DF['Gaussian Naive Bayes'][dist].append(Accuracy)



    X = np.array(Distance_DF[distance_list2])
    gnb = GaussianNB()
    GNB = gnb.fit(X[Train], encoded_Y[Train])
    Predicted = GNB.predict(X[Test])
    Actual = np.array(encoded_Y[Test])
    Accuracy = np.sum((Predicted == Actual) * 1.) / len(Predicted)
    Accuracy_DF['Gaussian Naive Bayes']['All DistancesG'].append(Accuracy)

    # DT
    model_tree = tree.DecisionTreeClassifier()
    Tree = model_tree.fit(X[Train], encoded_Y[Train])
    Predicted = Tree.predict(X[Test])
    Actual = np.array(encoded_Y[Test])
    Accuracy = np.sum((Predicted == Actual) * 1.) / len(Predicted)
    Accuracy_DF['Decision Tree']['All DistancesD'].append(Accuracy)

    # SVM
    model_SVM = svm.SVC(cache_size=4000, C = 0.4)
    SVM = model_SVM.fit(X[Train], encoded_Y[Train])
    Predicted = SVM.predict(X[Test])
    Actual = np.array(encoded_Y[Test])
    Accuracy = np.sum((Predicted == Actual) * 1.) / len(Predicted)
    Accuracy_DF['Support Vector Machine']['All DistancesS'].append(Accuracy)

    # NN
    epo = 800
    bs = 80

    ModelFit = create_model()
    ModelFit.fit(X[Train], encoded_Y[Train], epochs=epo, batch_size=bs, verbose=0)
    Predicted = ModelFit.predict(X[Test])
    Predicted = np.argmax(Predicted, axis = 1)
    Actual = np.array(encoded_Y[Test])
    Accuracy = np.sum((Predicted == Actual) * 1.) / len(Predicted)
    Accuracy_DF['Neural Network']['All DistancesN'].append(Accuracy)

# k fold testing to evaluate NN ability to predict Metrics from ss volume w scores
NN_DF = pd.merge(left=Vol_W_DF_Var, right=Distance_DF[['SubjID'] + Metrics], left_on='SubjID', right_on='SubjID')
NN_DF = NN_DF.reset_index(drop = True)
MSE_DF = {}
MSE_DF_Final = {}
for M in Metrics:
    MSE_DF[M] = []
    MSE_DF_Final[M] = ''
epo = 100
bs = 50
for hi in range(k):
    NT = [150, 200, 60]  # number of training patients from each group
    Train = random.sample(NN_DF[NN_DF['DX'] == 'CN'].index.tolist(), NT[0]) + random.sample(NN_DF[NN_DF['DX'] == 'MCI'].index.tolist(), NT[1]) + random.sample(NN_DF[NN_DF['DX'] == 'Dementia'].index.tolist(), NT[2])
    Test = [x for x in range(0, NN_DF.__len__()) if x not in Train]
    X = np.array(NN_DF[ss_regions]).astype(float)
    for met in Metrics:
        Y = np.array(NN_DF[met]).astype(float)
        ModelFit = create_model2()
        ModelFit.fit(X[Train], Y[Train], epochs=epo, batch_size=bs, verbose=0)
        Predicted = ModelFit.predict(X[Test])
        mse = np.power(Y[Test]-Predicted, 2).sum()/len(Y)
        MSE_DF[met].append(mse)


# Placing Results of K-fold testing in DFs
for disti in distance_list:
    dict_ML['Gaussian Naive Bayes'][disti] = str(np.array(Accuracy_DF['Gaussian Naive Bayes'][disti]).mean()) + " (" + str(np.array(Accuracy_DF['Gaussian Naive Bayes'][disti]).std()) + ") --" + str(np.array(Accuracy_DF['Gaussian Naive Bayes'][disti]).max())

dict_ML['Gaussian Naive Bayes']['All Distances'] = str(np.array(Accuracy_DF['Gaussian Naive Bayes']['All DistancesG']).mean()) + " (" + str(np.array(Accuracy_DF['Gaussian Naive Bayes']['All DistancesG']).std()) + ") --" + str(np.array(Accuracy_DF['Gaussian Naive Bayes']['All DistancesG']).max())
dict_ML['Decision Tree']['All Distances'] = str(np.array(Accuracy_DF['Decision Tree']['All DistancesD']).mean()) + " (" + str(np.array(Accuracy_DF['Decision Tree']['All DistancesD']).std()) + ") --" + str(np.array(Accuracy_DF['Decision Tree']['All DistancesD']).max())
dict_ML['Support Vector Machine']['All Distances'] = str(np.array(Accuracy_DF['Support Vector Machine']['All DistancesS']).mean()) + " (" + str(np.array(Accuracy_DF['Support Vector Machine']['All DistancesS']).std()) + ") --" + str(np.array(Accuracy_DF['Support Vector Machine']['All DistancesS']).max())
dict_ML['Neural Network']['All Distances'] = str(np.array(Accuracy_DF['Neural Network']['All DistancesN']).mean()) + " (" + str(np.array(Accuracy_DF['Neural Network']['All DistancesN']).std()) + ") --" + str(np.array(Accuracy_DF['Neural Network']['All DistancesN']).max())


for met in Metrics:
    MSE_DF_Final[met] = str(np.array(MSE_DF[met]).mean()) + " (" + str(np.array(MSE_DF[met]).std()) + ")"

MSE_DF_Final = {'MSE': MSE_DF_Final}

# Comparing correlations between AHV and ZWE
R2_Table = pd.DataFrame(R2_DF)
R2_Table_CN = pd.DataFrame(R2_DF_CN)
R2_Table_MCI = pd.DataFrame(R2_DF_MCI)
R2_Table_Dementia = pd.DataFrame(R2_DF_Dementia)
Corr_Comp = {}
for coll in R2_Table.columns:
    r1 = float(R2_Table[coll]['AHV'].split(" ")[0])
    n1 = float(R2_Table[coll]['AHV'].split(" ")[-1])
    r2 = float(R2_Table[coll]['ADNeuro Score'].split(" ")[0])
    n2 = float(R2_Table[coll]['ADNeuro Score'].split(" ")[-1])
    Corr_Comp[coll] = Compare_Corr(r1, r2, n1, n2)
R_Comp_Table = pd.DataFrame(Corr_Comp, index=[0])

Corr_Comp = {}
for coll in R2_Table_CN.columns:
    r1 = float(R2_Table_CN[coll]['AHV'].split(" ")[0])
    n1 = float(R2_Table_CN[coll]['AHV'].split(" ")[-1])
    r2 = float(R2_Table_CN[coll]['ADNeuro Score'].split(" ")[0])
    n2 = float(R2_Table_CN[coll]['ADNeuro Score'].split(" ")[-1])
    Corr_Comp[coll] = Compare_Corr(r1, r2, n1, n2)
R_Comp_Table_CN = pd.DataFrame(Corr_Comp, index=[0])

Corr_Comp = {}
for coll in R2_Table_MCI.columns:
    r1 = float(R2_Table_MCI[coll]['AHV'].split(" ")[0])
    n1 = float(R2_Table_MCI[coll]['AHV'].split(" ")[-1])
    r2 = float(R2_Table_MCI[coll]['ADNeuro Score'].split(" ")[0])
    n2 = float(R2_Table_MCI[coll]['ADNeuro Score'].split(" ")[-1])
    Corr_Comp[coll] = Compare_Corr(r1, r2, n1, n2)
R_Comp_Table_MCI = pd.DataFrame(Corr_Comp, index=[0])

Corr_Comp = {}
for coll in R2_Table_Dementia.columns:
    r1 = float(R2_Table_Dementia[coll]['AHV'].split(" ")[0])
    n1 = float(R2_Table_Dementia[coll]['AHV'].split(" ")[-1])
    r2 = float(R2_Table_Dementia[coll]['ADNeuro Score'].split(" ")[0])
    n2 = float(R2_Table_Dementia[coll]['ADNeuro Score'].split(" ")[-1])
    Corr_Comp[coll] = Compare_Corr(r1, r2, n1, n2)
R_Comp_Table_Dementia = pd.DataFrame(Corr_Comp, index=[0])
# Saving Results
ML_Table = pd.DataFrame(dict_ML)
P_Table = pd.DataFrame(pairwise_dict)
ROC_AUC_Table = pd.DataFrame(log_AUC_dict)
MSE_Table = pd.DataFrame(MSE_DF_Final)

Anova_Z = pd.DataFrame(anova_z_table)
ML_Table.to_excel(proj_dir + r'\ML_Table.xlsx')
P_Table.to_excel(proj_dir + r'\P_Table.xlsx')
ROC_AUC_Table.to_excel(proj_dir + r'\ROC_AUC_Table.xlsx')
R2_Table.to_excel(proj_dir + r'\R_Table.xlsx')
R2_Table_CN.to_excel(proj_dir + r'\R_Table_CN.xlsx')
R2_Table_MCI.to_excel(proj_dir + r'\R_Table_MCI.xlsx')
R2_Table_Dementia.to_excel(proj_dir + r'\R_Table_Dementia.xlsx')
MSE_Table.to_excel(proj_dir + r'\MSE_Table.xlsx')
Preg_DF.to_excel(proj_dir + r'\P_Reg_Table.xlsx')
Preg_DF_z2.to_excel(proj_dir + r'\Z_Reg_Table.xlsx')
Preg_DF_z.to_excel(proj_dir + r'\Z_Weights.xlsx')
R_Comp_Table.to_excel(proj_dir + r'\R_Comp_Table.xlsx')
R_Comp_Table_CN.to_excel(proj_dir + r'\R_Comp_Table_CN.xlsx')
R_Comp_Table_MCI.to_excel(proj_dir + r'\R_Comp_Table_MCI.xlsx')
R_Comp_Table_Dementia.to_excel(proj_dir + r'\R_Comp_Table_Dementia.xlsx')
Anova_Z.to_excel(proj_dir + r'\Anova_Z.xlsx')


# Plotting Heatmap of distance metrics grouped by DX
Distance_HM = Distance_DF.copy()
Distance_HM = pd.concat([Distance_HM[Distance_HM['DX'] == 'CN'], Distance_HM[Distance_HM['DX'] == 'MCI'], Distance_HM[Distance_HM['DX'] == 'Dementia']], ignore_index=True)[distance_list]
DHMNorm = (Distance_HM[distance_list]-Distance_HM[distance_list].mean())/Distance_HM[distance_list].std()
for h in DHMNorm.columns:
    DHMNorm[h] = DHMNorm[h].astype(float)

cg = sns.clustermap(pd.DataFrame(DHMNorm), yticklabels="", col_cluster=False, row_cluster=False)
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
cg.ax_row_dendrogram.set_xlim([0,0])
plt.show()

# Plotting Heatmap of P-values grouped by comparison
Log = -1*np.log10(Preg_DF.astype(float))
DHMNorm = (Preg_DF-Preg_DF.mean())/Preg_DF.std()
DHMNorm = Log

DHMNorm = (Log-Log.mean())/Log.std()
for h in DHMNorm.columns:
    DHMNorm[h] = DHMNorm[h].astype(float)

cg = sns.clustermap(pd.DataFrame(DHMNorm), figsize=(13, 20))
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
cg.ax_row_dendrogram.set_xlim([0,0])

plt.show()

################## Longitudinal Analysis #################
Prediction = ['DX']
Timelines = {}
Timelines_sp = {}
Comparisons = ['CN->CN vs CN->MCI', 'CN->CN vs CN->Dementia', 'CN->MCI vs CN->Dementia',
               'MCI->CN vs MCI->MCI', 'MCI->CN vs MCI->Dementia', 'MCI->MCI vs MCI->Dementia',
               'Dementia->CN vs Dementia->MCI', 'Dementia->CN vs Dementia->Dementia',
               'Dementia->MCI vs Dementia->Dementia']
Transitions = ['CN->CN', 'CN->MCI', 'CN->Dementia',   'MCI->CN', 'MCI->MCI', 'MCI->Dementia',   'Dementia->CN', 'Dementia->MCI', 'Dementia->Dementia']

DX_Map = {'CN': 0, 'MCI': 1, 'Dementia':2}

# Create DF for Long analysis of Youden index

matching = [s for s in Scan_Info_DF.columns if 'DX' in s] # Obtain all columns with substring and merge with distance metrics

Array1 = np.array(distance_list2)
Array2 = np.array(list(Youden_AUC_dict[Youden_AUC_dict.columns.to_list()[0]][Array1[0]].keys()))
ind_array = [Array1.repeat(len(Array2)), np.tile(Array2, len(Array1))]
Youden_AUC_DF_long = pd.DataFrame(columns=matching[1:], index=ind_array)
Youden_AUC_DF_long_CN = copy.deepcopy(Youden_AUC_DF_long)
Youden_AUC_DF_long_MCI = copy.deepcopy(Youden_AUC_DF_long)
Youden_AUC_DF_long_Dementia = copy.deepcopy(Youden_AUC_DF_long)

# Create DF for Long analysis of Cutoff index

matching = [s for s in Scan_Info_DF.columns if 'DX' in s] # Obtain all columns with substring and merge with distance metrics

Array1 = np.array(distance_list2)
Array2 = np.array(list(Cutoff_AUC_dict[Cutoff_AUC_dict.columns.to_list()[0]][Array1[0]].keys()))
ind_array = [Array1.repeat(len(Array2)), np.tile(Array2, len(Array1))]
Cutoff_AUC_DF_long = pd.DataFrame(columns=matching[1:], index=ind_array)

Cutoff_AUC_DF_long_CN = copy.deepcopy(Cutoff_AUC_DF_long)
Cutoff_AUC_DF_long_MCI = copy.deepcopy(Cutoff_AUC_DF_long)
Cutoff_AUC_DF_long_Dementia = copy.deepcopy(Cutoff_AUC_DF_long)

for P in Prediction:
    matching = [s for s in Scan_Info_DF.columns if P in s] # Obtain all columns with substring and merge with distance metrics
    DF = pd.merge(left=Distance_DF[['SubjID'] + distance_list2], right=Scan_Info_DF[['SubjID'] + matching], left_on='SubjID', right_on='SubjID')

    # Gross Change
    Timelines[P] = {}
    P_values = {}
    P_values_CN = {}
    P_values_MCI = {}
    P_values_Dementia = {}
    log_AUC_dict_long = {}
    log_AUC_dict_long_CN = {}
    log_AUC_dict_long_MCI = {}
    log_AUC_dict_long_Dementia = {}
    P_values_g = {}
    for m in matching[1:-1]:
        Timelines[P][m] = {}
        P_values[m] = {}
        P_values_CN[m] = {}
        P_values_MCI[m] = {}
        P_values_Dementia[m] = {}
        log_AUC_dict_long[m] = {}
        log_AUC_dict_long_CN[m] = {}
        log_AUC_dict_long_MCI[m] = {}
        log_AUC_dict_long_Dementia[m] = {}
        P_values_g[m] = {}
        DF_Temp = DF[['SubjID'] + [matching[0]] + [m]].dropna()
        DF_Temp = DF_Temp[(DF_Temp[matching[0]] != 0) & (DF_Temp[m] !=0) ]
        #Change_array = (DF_Temp[matching[0]] == DF_Temp[m]) == False
        mapped_BL = []
        mapped_TP = []
        for iii in DF_Temp[matching[0]]:
            mapped_BL.append(DX_Map[iii])
        for iii in DF_Temp[m]:
            mapped_TP.append(DX_Map[iii])


        Change_array = (pd.DataFrame(mapped_BL)<pd.DataFrame(mapped_TP))

        Timelines[P][m]['Worse'] = Change_array.value_counts()[True]
        Timelines[P][m]['Not Worse'] = Change_array.value_counts()[False]
        Timelines[P][m]['Total'] = Timelines[P][m]['Worse'] + Timelines[P][m]['Not Worse']
        for d in distance_list2:
            DF_Temp = DF[['SubjID'] + [matching[0]] + [m]  + [d]].dropna()
            DF_Temp = DF_Temp[(DF_Temp[matching[0]] != 0) & (DF_Temp[m] != 0)]
            mapped_BL = []
            mapped_TP = []
            for iii in DF_Temp[matching[0]]:
                mapped_BL.append(DX_Map[iii])
            for iii in DF_Temp[m]:
                mapped_TP.append(DX_Map[iii])
            Change_array = (pd.DataFrame(mapped_BL) < pd.DataFrame(mapped_TP)).to_numpy().flatten()
            Change_DF = pd.DataFrame({'SubjID': DF_Temp['SubjID'].to_list(), 'DX':DF_Temp['DX'], 'Distance': DF_Temp[d].to_list(), 'Worse': Change_array.tolist()})
            P_val = scipy.stats.ttest_ind(Change_DF['Distance'][Change_DF ['Worse'] == True], Change_DF ['Distance'][Change_DF ['Worse'] == False]).pvalue
            P_val_CN = scipy.stats.ttest_ind(Change_DF['Distance'][Change_DF['DX'] == 'CN'][Change_DF ['Worse'] == True], Change_DF ['Distance'][Change_DF['DX'] == 'CN'][Change_DF ['Worse'] == False]).pvalue
            P_val_MCI = scipy.stats.ttest_ind(Change_DF['Distance'][Change_DF['DX']=='MCI'][Change_DF ['Worse'] == True], Change_DF ['Distance'][Change_DF['DX'] == 'MCI'][Change_DF ['Worse'] == False]).pvalue
            P_val_Dementia = scipy.stats.ttest_ind(Change_DF['Distance'][Change_DF['DX']=='Dementia'][Change_DF ['Worse'] == True], Change_DF ['Distance'][Change_DF['DX'] == 'Dementia'][Change_DF ['Worse'] == False]).pvalue

            n1 = Change_DF['Distance'][Change_DF['Worse'] == True].to_list().__len__()
            n2 = Change_DF['Distance'][Change_DF['Worse'] == False].to_list().__len__()
            n1_CN = Change_DF['Distance'][Change_DF['DX']=='CN'][Change_DF['Worse'] == True].to_list().__len__()
            n2_CN = Change_DF['Distance'][Change_DF['DX']=='CN'][Change_DF['Worse'] == False].to_list().__len__()
            n1_MCI = Change_DF['Distance'][Change_DF['DX']=='MCI'][Change_DF['Worse'] == True].to_list().__len__()
            n2_MCI = Change_DF['Distance'][Change_DF['DX']=='MCI'][Change_DF['Worse'] == False].to_list().__len__()
            n1_Dementia = Change_DF['Distance'][Change_DF['DX']=='Dementia'][Change_DF['Worse'] == True].to_list().__len__()
            n2_Dementia = Change_DF['Distance'][Change_DF['DX']=='Dementia'][Change_DF['Worse'] == False].to_list().__len__()
            dd = pg.compute_effsize(Change_DF['Distance'][Change_DF['Worse'] == True].to_list(), Change_DF['Distance'][Change_DF['Worse'] == False].to_list(), eftype='cohen')
            dd_CN = pg.compute_effsize(Change_DF['Distance'][(Change_DF['Worse'] == True) & (Change_DF['DX']=='CN')].to_list(), Change_DF['Distance'][(Change_DF['Worse'] == False) & (Change_DF['DX']=='CN')].to_list(), eftype='cohen')
            dd_MCI = pg.compute_effsize(Change_DF['Distance'][(Change_DF['Worse'] == True) & (Change_DF['DX']=='MCI')].to_list(), Change_DF['Distance'][(Change_DF['Worse'] == False) & (Change_DF['DX']=='MCI')].to_list(), eftype='cohen')
            dd_Dementia = pg.compute_effsize(Change_DF['Distance'][(Change_DF['Worse'] == True)].to_list(), Change_DF['Distance'][(Change_DF['Worse'] == False)].to_list(), eftype='cohen')
            v1 = (((n1 + n2) / (n1 * n2) + dd ** 2 * (2 * (n1 + n2 - 2)) ** -1) * ((n1 + n2) / (n1 + n2 - 2))) ** 0.5
            v1_CN = (((n1_CN + n2_CN) / (n1_CN * n2_CN) + dd_CN ** 2 * (2 * (n1_CN + n2_CN - 2)) ** -1) * ((n1_CN + n2_CN) / (n1_CN + n2_CN - 2))) ** 0.5
            v1_MCI = (((n1_MCI + n2_MCI) / (n1_MCI * n2_MCI) + dd_MCI ** 2 * (2 * (n1_MCI + n2_MCI - 2)) ** -1) * ((n1_MCI + n2_MCI) / (n1_MCI + n2_MCI - 2))) ** 0.5

            r1 = str(dd - 1.96 * v1) + "-" + str(dd + 1.96 * v1)
            r1_CN = str(dd_CN - 1.96 * v1_CN) + "-" + str(dd_CN + 1.96 * v1_CN)
            r1_MCI = str(dd_MCI - 1.96 * v1_MCI) + "-" + str(dd_MCI + 1.96 * v1_MCI)


            P_values[m][d] = str(P_val) +" (" + str(scipy.stats.norm.ppf(P_val/2)) +")" + " d = " + str(dd) + " (" + r1 + ")"
            P_values_CN[m][d] = str(P_val_CN) + " (" + str(scipy.stats.norm.ppf(P_val_CN/ 2)) + ")" + " d = " + str(dd_CN) + " (" + r1_CN + ")"
            P_values_MCI[m][d] = str(P_val_MCI) + " (" + str(scipy.stats.norm.ppf(P_val_MCI / 2)) + ")" + " d = " + str(dd_MCI) + " (" + r1_MCI + ")"

            P_values_g[m][d] = -1*scipy.stats.norm.ppf(P_val/2)


            # ROC AUC
            for thing in range(3):
                lr_auc_list = []
                fp_list4 = []
                tp_list4 = []
                for i in range(kf):
                    if thing == 0:
                        y = np.array(Change_DF['Worse'].copy())
                        X = np.array(Change_DF['Distance'].copy())
                    if thing == 1:
                        y = np.array(Change_DF['Worse'][Change_DF['DX']=='CN'].copy())
                        X = np.array(Change_DF['Distance'][Change_DF['DX']=='CN'].copy())
                    if thing == 2:
                        y = np.array(Change_DF['Worse'][Change_DF['DX']=='MCI'].copy())
                        X = np.array(Change_DF['Distance'][Change_DF['DX']=='MCI'].copy())
                    if thing == 3:
                        y = np.array(Change_DF['Worse'][Change_DF['DX']=='Dementia'].copy())
                        X = np.array(Change_DF['Distance'][Change_DF['DX']=='Dementia'].copy())
                    trainX, testX, trainy, testy = tt_split(X, y, train_size=ts)
                    model = LogisticRegression(solver='lbfgs')
                    model.fit(np.array(trainX).reshape(-1, 1), np.array(trainy))
                    lr_probs = model.predict_proba(np.array(testX).reshape(-1, 1))
                    lr_probs = lr_probs[:, 1]
                    lr_auc = roc_auc_score(testy, lr_probs)
                    fpr, tpr, thresholds = roc_curve(testy, lr_probs, pos_label=True, drop_intermediate=False)
                    lr_auc_list.append(lr_auc)
                    fp_list4.append(fpr)
                    tp_list4.append(tpr)

                meann = np.array(lr_auc_list).mean()
                std = np.array(lr_auc_list).std()
                if thing == 0:
                    log_AUC_dict_long[m][d] = str(meann) + " (" + str(meann - 1.96 * std) + "-" + str(
                        meann + 1.96 * std) + ")"
                if thing == 1:
                    log_AUC_dict_long_CN[m][d] = str(meann) + " (" + str(meann - 1.96 * std) + "-" + str(
                        meann + 1.96 * std) + ")"
                if thing == 2:
                    log_AUC_dict_long_MCI[m][d] = str(meann) + " (" + str(meann - 1.96 * std) + "-" + str(
                        meann + 1.96 * std) + ")"
                if thing == 3:
                    log_AUC_dict_long_Dementia[m][d] = str(meann) + " (" + str(meann - 1.96 * std) + "-" + str(
                        meann + 1.96 * std) + ")"
                tp_list4 = [x for x in tp_list4 if len(x) == st.mode([len(x) for x in tp_list4])]
                fp_list4 = [x for x in fp_list4 if len(x) == st.mode([len(x) for x in fp_list4])]
                tp_m = np.stack(tp_list4).mean(axis=0)
                fp_m = np.stack(fp_list4).mean(axis=0)
                indss, dddi = Youden_index_analysis(tp_m, fp_m)
                indsss, dddii = Cutoff_index_analysis(tp_m, fp_m)

                if thing == 0:
                    for kk in dddi.keys():
                        Youden_AUC_DF_long.loc[(d, kk), m] = dddi[kk]
                    for kk in dddii.keys():
                        Cutoff_AUC_DF_long.loc[(d, kk), m] = dddii[kk]
                if thing == 1:
                    for kk in dddi.keys():
                        Youden_AUC_DF_long_CN.loc[(d, kk), m] = dddi[kk]
                    for kk in dddii.keys():
                        Cutoff_AUC_DF_long_CN.loc[(d, kk), m] = dddii[kk]
                if thing == 2:
                    for kk in dddi.keys():
                        Youden_AUC_DF_long_MCI.loc[(d, kk), m] = dddi[kk]
                    for kk in dddii.keys():
                        Cutoff_AUC_DF_long_MCI.loc[(d, kk), m] = dddii[kk]
                if thing == 3:
                    for kk in dddi.keys():
                        Youden_AUC_DF_long_Dementia.loc[(d, kk), m] = dddi[kk]
                    for kk in dddii.keys():
                        Cutoff_AUC_DF_long_Dementia.loc[(d, kk), m] = dddii[kk]


    #Specific Change
    Timelines_sp[P] = {}
    P_values_sp = {}
    P_values_sp_g = {}
    Means_sp = {}
    for m in matching[1:]:
        Timelines_sp[P][m] = {}
        P_values_sp[m] = {}
        P_values_sp_g[m] = {}
        Means_sp[m] = {}
        DF_Temp = DF[['SubjID'] + [matching[0]] + [m]].dropna()
        DF_Temp = DF_Temp[(DF_Temp[matching[0]] != 0) & (DF_Temp[m] !=0) ]
        Change_array_sp = DF_Temp[matching[0]].astype(str) + "->" + DF_Temp[m].astype(str)

        Timelines_sp[P][m]['Change'] = dict(Change_array_sp.value_counts())
        Timelines_sp[P][m]['Total'] = Change_array_sp.value_counts().sum()
        for d in distance_list2:
            P_values_sp[m][d] = {}
            Means_sp[m][d] = {}
            DF_Temp = DF[['SubjID'] + [matching[0]] + [m] + [d]].dropna()
            DF_Temp = DF_Temp[(DF_Temp[matching[0]] != 0) & (DF_Temp[m] != 0)]
            Change_array_sp = DF_Temp[matching[0]].astype(str) + "->" + DF_Temp[m].astype(str)
            Change_DF_sp = pd.DataFrame({'SubjID': DF_Temp['SubjID'].to_list(), 'Distance': DF_Temp[d].to_list(), 'Change': Change_array_sp.to_list()})

            Comp_DF = {}
            Comp_DF_g = {}
            for cmp in Comparisons:
                g1, g2 = cmp.split(" vs ")
                P_val = scipy.stats.ttest_ind(Change_DF_sp['Distance'][Change_DF_sp['Change'] == g1],Change_DF_sp['Distance'][Change_DF_sp['Change'] == g2]).pvalue
                if np.isnan(P_val) == False:
                    n1 = Change_DF_sp['Distance'][Change_DF_sp['Change'] == g1].to_list().__len__()
                    n2 = Change_DF_sp['Distance'][Change_DF_sp['Change'] == g2].to_list().__len__()
                    dd = pg.compute_effsize(Change_DF_sp['Distance'][Change_DF_sp['Change'] == g1].to_list(), Change_DF_sp['Distance'][Change_DF_sp['Change'] == g2].to_list(), eftype='cohen')
                    v1 = (((n1 + n2) / (n1 * n2) + dd ** 2 * (2 * (n1 + n2 - 2)) ** -1) * ((n1 + n2) / (n1 + n2 - 2))) ** 0.5
                    r1 = str(dd - 1.96 * v1) + "-" + str(dd + 1.96 * v1)

                    Comp_DF[cmp] = str(P_val) + " (" + str(scipy.stats.norm.ppf(P_val/2)) + ")" + " d = " + str(dd) + " (" + r1 + ")"
                    Comp_DF_g[cmp] = -1*scipy.stats.norm.ppf(P_val/2)
            P_values_sp[m][d] = Comp_DF
            P_values_sp_g[m][d] = Comp_DF_g

            for T in Transitions:
                mn = Change_DF_sp['Distance'][Change_DF_sp['Change'] == T].mean()
                std = Change_DF_sp['Distance'][Change_DF_sp['Change'] == T].std()
                z = 0.524401
                Means_sp[m][d][T] = str(mn) + " (" + str(std) + ") [" + str(mn - std*z) +"-" +str(mn + std*z)+"]"


# Longitudinal prediction of change in Metrics

# Create DF to hold Data

n_list = []
for Me in Metrics:
    matching1 = [s for s in Scan_Info_DF.columns if Me in s]
    numbers = [n.split("m")[1] for n in matching1[1:]]
    for n in numbers:
        if (n in n_list) == False:
            n_list.append(n)
n_list = sorted(n_list, key = int)

d_array = np.array(distance_list2)
Me_array = np.array(Metrics)
ind_array = [d_array.repeat(len(Me_array)), np.tile(Me_array,len(d_array))]

Long_R_DF = pd.DataFrame(columns=n_list, index=ind_array)
Long_R_DF_CN = pd.DataFrame(columns=n_list, index=ind_array)
Long_R_DF_MCI = pd.DataFrame(columns=n_list, index=ind_array)
Long_R_DF_Dementia = pd.DataFrame(columns=n_list, index=ind_array)

# fill DF with R value
for Me in Metrics:
    matching1 = [s for s in Scan_Info_DF.columns if Me in s]
    DF12 = pd.merge(left=Distance_DF[['SubjID'] + ['DX'] +distance_list2], right=Scan_Info_DF[['SubjID'] + matching1], left_on='SubjID', right_on='SubjID')
    for d in distance_list2:
        for m in matching1[1:]:
            DF_Temp = DF12[['SubjID', 'DX'] + [matching1[0]] + [m] + [d]].dropna()
            DF_Temp = DF_Temp[(DF_Temp[matching1[0]] != 0) & (DF_Temp[m] != 0)]
            Change_array_sp = DF_Temp[matching1[0]] - DF_Temp[m]
            Change_DF_sp = pd.DataFrame({'SubjID': DF_Temp['SubjID'].to_list(), 'DX': DF_Temp['DX'].to_list(), 'Distance': DF_Temp[d].to_list(), 'Change': Change_array_sp.to_list()})
            res = scipy.stats.linregress(Change_DF_sp['Distance'].to_list(), Change_DF_sp['Change'].to_list())
            Long_R_DF.loc[d, Me][m.split("m")[1]] = str(res.rvalue) + " (n = " + str(Change_DF_sp['Change'].to_list().__len__()) + ")" + " (p = " + str(res.pvalue) + ")"

            if Change_DF_sp[Change_DF_sp['DX'] == 'CN'].__len__() !=0:
                res_CN = scipy.stats.linregress(Change_DF_sp[Change_DF_sp['DX'] == 'CN']['Distance'].to_list(), Change_DF_sp[Change_DF_sp['DX'] == 'CN']['Change'].to_list())
                Long_R_DF_CN.loc[d, Me][m.split("m")[1]] = str(res_CN.rvalue) + " (n = " + str(Change_DF_sp[Change_DF_sp['DX'] == 'CN']['Change'].to_list().__len__()) + ")" + " (p = " + str(res_CN.pvalue) + ")"
            else:
                Long_R_DF_CN.loc[d, Me][m.split("m")[1]] = 'NA' + " (n = " + str(Change_DF_sp[Change_DF_sp['DX'] == 'CN']['Change'].to_list().__len__()) + ")" + " (p = " + 'NA' + ")"
            if Change_DF_sp[Change_DF_sp['DX'] == 'MCI'].__len__() !=0:
                res_MCI = scipy.stats.linregress(Change_DF_sp[Change_DF_sp['DX'] == 'MCI']['Distance'].to_list(), Change_DF_sp[Change_DF_sp['DX'] == 'MCI']['Change'].to_list())
                Long_R_DF_MCI.loc[d, Me][m.split("m")[1]] = str(res_MCI.rvalue) + " (n = " + str(Change_DF_sp[Change_DF_sp['DX'] == 'MCI']['Change'].to_list().__len__()) + ")" + " (p = " + str(res_MCI.pvalue) + ")"
            else:
                Long_R_DF_MCI.loc[d, Me][m.split("m")[1]] = 'NA' + " (n = " + str(Change_DF_sp[Change_DF_sp['DX'] == 'MCI']['Change'].to_list().__len__()) + ")" + " (p = " + 'NA' + ")"

            if Change_DF_sp[Change_DF_sp['DX'] == 'Dementia'].__len__() !=0:
                res_Dementia = scipy.stats.linregress(Change_DF_sp[Change_DF_sp['DX'] == 'Dementia']['Distance'].to_list(), Change_DF_sp[Change_DF_sp['DX'] == 'Dementia']['Change'].to_list())
                Long_R_DF_Dementia.loc[d, Me][m.split("m")[1]] = str(res_Dementia.rvalue) + " (n = " + str(Change_DF_sp[Change_DF_sp['DX'] == 'Dementia']['Change'].to_list().__len__()) + ")" + " (p = " + str(res_Dementia.pvalue) + ")"
            else:
                Long_R_DF_Dementia.loc[d, Me][m.split("m")[1]] = 'NA' + " (n = " + str(Change_DF_sp[Change_DF_sp['DX'] == 'Dementia']['Change'].to_list().__len__()) + ")" + " (p = " + 'NA' + ")"


# Saving Results
PV_Table = pd.DataFrame(P_values)
PV_Table_CN = pd.DataFrame(P_values_CN)
PV_Table_MCI = pd.DataFrame(P_values_MCI)
PV_Table_Dementia = pd.DataFrame(P_values_Dementia)
Timeline_Table = pd.DataFrame(Timelines)
log_AUC_dict_long_T = pd.DataFrame(log_AUC_dict_long)
log_AUC_dict_long_T_CN = pd.DataFrame(log_AUC_dict_long_CN)
log_AUC_dict_long_T_MCI = pd.DataFrame(log_AUC_dict_long_MCI)
log_AUC_dict_long_T_Dementia = pd.DataFrame(log_AUC_dict_long_Dementia)

Long_R_DF.to_excel(proj_dir + r'\Long_R_Table.xlsx')
Long_R_DF_CN.to_excel(proj_dir + r'\Long_R_Table_CN.xlsx')
Long_R_DF_MCI.to_excel(proj_dir + r'\Long_R_Table_MCI.xlsx')
Long_R_DF_Dementia.to_excel(proj_dir + r'\Long_R_Table_Dementia.xlsx')
PV_Table.to_excel(proj_dir + r'\Long_P_Table.xlsx')
PV_Table_CN.to_excel(proj_dir + r'\Long_P_Table_CN.xlsx')
PV_Table_MCI.to_excel(proj_dir + r'\Long_P_Table_MCI.xlsx')
PV_Table_Dementia.to_excel(proj_dir + r'\Long_P_Table_Dementia.xlsx')
Timeline_Table.to_excel(proj_dir + r'\Long_Timeline_Table.xlsx')
log_AUC_dict_long_T.to_excel(proj_dir + r'\Long_ROC_AUC_Table.xlsx')
log_AUC_dict_long_T_CN.to_excel(proj_dir + r'\Long_ROC_AUC_Table_CN.xlsx')
log_AUC_dict_long_T_MCI.to_excel(proj_dir + r'\Long_ROC_AUC_Table_MCI.xlsx')
log_AUC_dict_long_T_Dementia.to_excel(proj_dir + r'\Long_ROC_AUC_Table_Dementia.xlsx')
Youden_AUC_DF_long.to_excel(proj_dir + r'\Long_Youden_AUC_DF_Table.xlsx')
Youden_AUC_DF_long_CN.to_excel(proj_dir + r'\Long_Youden_AUC_DF_Table_CN.xlsx')
Youden_AUC_DF_long_MCI.to_excel(proj_dir + r'\Long_Youden_AUC_DF_Table_MCI.xlsx')
Youden_AUC_DF_long_Dementia.to_excel(proj_dir + r'\Long_Youden_AUC_DF_Table_Dementia.xlsx')
Cutoff_AUC_DF_long.to_excel(proj_dir + r'\Long_Cutoff_AUC_DF_Table.xlsx')
Cutoff_AUC_DF_long_CN.to_excel(proj_dir + r'\Long_Cutoff_AUC_DF_Table_CN.xlsx')
Cutoff_AUC_DF_long_MCI.to_excel(proj_dir + r'\Long_Cutoff_AUC_DF_Table_MCI.xlsx')
Cutoff_AUC_DF_long_Dementia.to_excel(proj_dir + r'\Long_Cutoff_AUC_DF_Table_dementia.xlsx')

Long_R_Comp_DF = pd.DataFrame(columns=Long_R_DF.columns, index=Me_array)
Long_R_Comp_DF_CN = Long_R_Comp_DF.copy()
Long_R_Comp_DF_MCI = Long_R_Comp_DF.copy()
Long_R_Comp_DF_Dementia = Long_R_Comp_DF.copy()

Comp_DF_list = [Long_R_Comp_DF, Long_R_Comp_DF_CN, Long_R_Comp_DF_MCI, Long_R_Comp_DF_Dementia]
O_DF_list = [Long_R_DF, Long_R_DF_CN, Long_R_DF_MCI, Long_R_DF_Dementia]

for i, DF1 in enumerate(Comp_DF_list):
    DF2 = O_DF_list[i]
    for col in DF1.columns:
        for me in Me_array:
            if (str(DF2.loc['AHV', me][col])=='nan' or str(DF2.loc['ADNeuro Score', me][col])=='nan') or (str(DF2.loc['AHV', me][col]).split(" ")[0]=='NA' or str(DF2.loc['ADNeuro Score', me][col]).split(" ")[0]=='NA'):
                DF1[col][me] = "N/A"
            else:
                r1 = float(DF2.loc['AHV', me][col].split(" ")[0])
                r2 = float(DF2.loc['ADNeuro Score', me][col].split(" ")[0])
                n1 = float(DF2.loc['AHV', me][col].split("n = ")[1].split(")")[0])
                n2 = float(DF2.loc['ADNeuro Score', me][col].split("n = ")[1].split(")")[0])
                DF1[col][me] = Compare_Corr(r1, r2, n1, n2)

Long_R_Comp_DF.to_excel(proj_dir + r'\Long_R_Comp_Table.xlsx')
Long_R_Comp_DF_CN.to_excel(proj_dir + r'\Long_R_Comp_Table_CN.xlsx')
Long_R_Comp_DF_MCI.to_excel(proj_dir + r'\Long_R_Comp_Table_MCI.xlsx')
Long_R_Comp_DF_Dementia.to_excel(proj_dir + r'\Long_R_Comp_Table_Dementia.xlsx')

#knitting HTML of R data
DF_list = [Long_R_DF.copy(), Long_R_DF_CN.copy(), Long_R_DF_MCI.copy(), Long_R_DF_Dementia.copy()]
a = ['','CN','MCI', 'Dementia']

for cc, s in enumerate(DF_list):
    s = s.style
    font = 'Arial'
    cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'td:hover',
        'props': [('background-color', '#ffff0066')]
    }
    index_names = {
        'selector': '.index_name',
        'props': 'color: darkgrey; font-weight:normal;  font-family:' + font + ';'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #1414144d; color: Black; font-family:' + font + ';'
    }
    s.set_table_styles([cell_hover, index_names, headers])

    idx = pd.IndexSlice

    s.set_caption('Longitudinal Linear Regression Results').set_table_styles([{'selector': 'caption',
                                            'props': 'caption-side: top; font-size:1.25em; background-color: #141478b3; color: white; font-family:' + font + ';'}],
                                          overwrite=False)
    s = s.to_html()
    s = s.replace("->", "&#8594")

    f = open(proj_dir + '\\' + 'Long_R_Table' + a[cc] + '.html', 'w')
    f.write(s)
    f.close()


#Creating Figures
P_values_g_DF = pd.DataFrame(P_values_g)
cols = P_values_g_DF.columns
res = [int(sub.split('m')[1]) for sub in cols]
P_values_g_DF.columns = res
P_values_g_DF = P_values_g_DF.transpose()
sns.set_style("white")
ax = P_values_g_DF.plot(y = distance_list2, kind="bar", colormap = 'Dark2')
ax.set_ylabel("-Z Transformed P Value")
ax.set_xlabel("Time after Baseline Diagnosis (Months)")
plt.show()


fig, ax = plt.subplots(3, 3, figsize = (15, 15))
for i, cmp in enumerate(Comparisons):
    ci = int(np.floor(i/3))
    cj = i%3
    P_values_sp_g_DF = pd.DataFrame(P_values_sp_g)

    for col in P_values_sp_g_DF.columns:
        for ii in range(P_values_sp_g_DF.__len__()):
            try:
                P_values_sp_g_DF[col][ii] = P_values_sp_g_DF[col][ii][cmp]
            except KeyError:
                P_values_sp_g_DF[col][ii] = np.nan


    cols = P_values_sp_g_DF.columns
    res = [int(sub.split('m')[1]) for sub in cols]
    P_values_sp_g_DF.columns = res
    P_values_sp_g_DF = P_values_sp_g_DF.transpose()
    sns.set_style("white")
    P_values_sp_g_DF.plot(y=distance_list2, kind="bar", colormap='Dark2', ax = ax[ci, cj])
    ax[ci, cj].set_ylabel("-Z Transformed P Value for " + cmp)
    ax[ci, cj].set_xlabel("Time after Baseline Diagnosis (Months)")
plt.show()

#Prediction Tool

def Predict_Trans_Perc(d_val: float, distance: str, timepoint: str):
    result = {}
    trans = Means_sp[timepoint][distance]
    for k in trans.keys():
        v = trans[k]
        mnn = float(v.split(" (")[0])
        stdd = float(v.split("(")[1].split(")")[0])
        if np.isnan(mnn) == False:
            zs = abs(d_val-mnn)/stdd
            result[k] = scipy.stats.norm.cdf(zs)
    return result


# Prediction Report
Perc_rep = {}

Pt_id = '130_S_2373'
for d in distance_list2:
    Perc_rep[d] = {}
    for m in matching[1:]:
        DF_Temp = DF[['SubjID'] + [matching[0]] + [d] + [m]].dropna()
        DF_Temp = DF_Temp[(DF_Temp[matching[0]] != 0) & (DF_Temp[m] !=0) ]
        pt = DF_Temp[DF_Temp['SubjID'] == Pt_id]
        if not list(pt.index):
            pass
        else:
            a = Predict_Trans_Perc(pt[d], d, m)
            rel_keys = [s for s in a.keys() if s.split('->')[0] == str(pt['DX'].values[0])]
            a = pd.DataFrame(a)[rel_keys]
            #.to_dict(orient = "records")[0]
            Perc_rep[d][m] = a
            Perc_rep[d][m]['True'] = pt['DX'].values[0] + '->' + pt[m].values[0]

# Styling and Saving Report
for i in Perc_rep:
    df1 = pd.concat(Perc_rep[i].values(), ignore_index=True, keys=Perc_rep[i].keys())
    df1['Time Point'] = Perc_rep[i].keys()
    df1 = df1.set_index('Time Point')
    Perc_rep[i] = df1.transpose()

Perc_rep = pd.concat(Perc_rep.values(), keys=Perc_rep.keys())

s = pd.DataFrame(Perc_rep)
res = [sub.split('m')[1] for sub in s.columns]
res = [r + " Months" for r in res]
s.columns = res
s.to_excel(proj_dir + r'\Long_Percentile_Report.xlsx')
s = s.style
font = 'Arial'
cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffff0066')]
}
index_names = {
    'selector': '.index_name',
    'props':  'color: darkgrey; font-weight:normal;  font-family:' + font + ';'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #1414144d; color: Black; font-family:' + font + ';'
}
s.set_table_styles([cell_hover, index_names, headers])

idx = pd.IndexSlice
slice_ = idx[idx[:,'True'], idx['12 Months':'96 Months']]
s.set_properties(**{'background-color': '#0096004d'}, subset=slice_)

s.set_caption('Percentile Report for patient ' +  Pt_id).set_table_styles([{'selector': 'caption','props': 'caption-side: top; font-size:1.25em; background-color: #141478b3; color: white; font-family:' + font + ';'}], overwrite=False)

s = s.to_html()
s = s.replace("->", "&#8594")
f = open(proj_dir + r'\Longitudinal Percentile Report.html', 'w')
f.write(s)
f.close()

#Styling and Saving Other Data


def Style_and_Save(DF_in: pd.DataFrame, Title: str):

    for i in DF_in:
        for j in DF_in[i]:
            if Title == 'Long_Distance_Ranges':
                for k in DF_in[i][j]:
                    if DF_in[i][j][k].__contains__('nan'):
                        DF_in[i][j][k] = 'Not Enough Data'
                    else:
                        DF_in[i][j][k] = DF_in[i][j][k].split("[")[1].replace("]","")
                        vali = DF_in[i][j][k]
                        vali = vali.split("-")
                        for ll, l in enumerate(vali):
                            if float("{:.2e}".format(float(l)).split("e")[1])<0:
                                vali[ll] = "{:.2e}".format(float(l))
                            else:
                                vali[ll] = "{:.2f}".format(float(l))
                        vali = "-".join(vali)
                        DF_in[i][j][k] = vali
                DF_in[i][j] = pd.DataFrame.from_records([DF_in[i][j]])
            else:
                DF_in[i][j] = pd.DataFrame.from_records([DF_in[i][j]])

    for i in DF_in:
        df1 = pd.concat(DF_in[i].values(), ignore_index=True, keys=DF_in[i].keys())
        df1[' '] = DF_in[i].keys()
        df1 = df1.set_index(' ')
        DF_in[i] = df1.transpose()

    DF_in = pd.concat(DF_in.values(), keys=DF_in.keys())

    s = pd.DataFrame(DF_in).copy()
    if Title != 'Long_Percent_Correct':
        res = [sub.split('m')[1] for sub in s.index.levels[0]]
        res = [r + " Months" for r in res]
        new_inds = tuple(s.index.set_levels(res, level = 0))
        s = s.transpose()
        s.columns = new_inds
        s = s.transpose()
    s.to_excel(proj_dir + '\\' + Title + '.xlsx')


    s = s.style
    font = 'Arial'
    cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'td:hover',
        'props': [('background-color', '#ffff0066')]
    }
    index_names = {
        'selector': '.index_name',
        'props':  'color: darkgrey; font-weight:normal;  font-family:' + font + ';'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #1414144d; color: Black; font-family:' + font + ';'
    }
    s.set_table_styles([cell_hover, index_names, headers])

    idx = pd.IndexSlice

    s.set_caption(Title).set_table_styles([{'selector': 'caption','props': 'caption-side: top; font-size:1.25em; background-color: #141478b3; color: white; font-family:' + font + ';'}], overwrite=False)
    s = s.to_html()
    s = s.replace("->", "&#8594")

    f = open(proj_dir + '\\' + Title + '.html', 'w')
    f.write(s)
    f.close()


Style_and_Save(copy.deepcopy(Means_sp), 'Long_Distance_Ranges')

Style_and_Save(copy.deepcopy(P_values_sp), 'Long_P_Table_sp')
DF_in = copy.deepcopy(Timelines_sp)
Title = 'Long_Timeline_Table_sp'

Timelines_sp = Timelines_sp['DX']
for i in Timelines_sp:
    asw = Timelines_sp[i]['Change']
    asw['Total'] = Timelines_sp[i]['Total']
    Timelines_sp[i] = asw

Timeline_Table_sp = pd.DataFrame(Timelines_sp)
Timeline_Table_sp.to_excel(proj_dir + r'\Long_Timeline_Table_sp.xlsx')

# Testing Percentile Predictor

res = [sub.split('m')[1] for sub in matching[1:]]
res = [r + " Months" for r in res]

Perf = {}
for d in distance_list2:
    Perf[d] = {}
    for kl in res:
        Perf[d][kl] = {'Correct': 0, 'Incorrect': 0, 'Percent correct': 0}

for Pt_id in tqdm(DF['SubjID']):
    subje = DF[['SubjID'] + matching[1:]][DF['SubjID'] == Pt_id].dropna(axis=1)
    list_to_drop = []
    for xx in subje.columns:
        if subje[xx].values[0] == 0:
            list_to_drop.append(xx)
    subje = subje.drop(columns=list_to_drop)
    if list(subje.columns) == ['SubjID']:
        pass
    else:
        Perc_rep = {}
        for d in distance_list2:
            Perc_rep[d] = {}
            for m in matching[1:]:
                DF_Temp = DF[['SubjID'] + [matching[0]] + [d] + [m]].dropna()
                DF_Temp = DF_Temp[(DF_Temp[matching[0]] != 0) & (DF_Temp[m] !=0) ]
                pt = DF_Temp[DF_Temp['SubjID'] == Pt_id]
                if not list(pt.index):
                    pass
                else:
                    a = Predict_Trans_Perc(pt[d], d, m)
                    rel_keys = [s for s in a.keys() if s.split('->')[0] == str(pt['DX'].values[0])]
                    a = pd.DataFrame(a)[rel_keys]
                    Perc_rep[d][m] = a
                    Perc_rep[d][m]['True'] = pt['DX'].values[0] + '->' + pt[m].values[0]


        for i in Perc_rep:
            df1 = pd.concat(Perc_rep[i].values(), ignore_index=True, keys=Perc_rep[i].keys())
            df1['Time Point'] = Perc_rep[i].keys()
            df1 = df1.set_index('Time Point')
            Perc_rep[i] = df1.transpose()

        Perc_rep = pd.concat(Perc_rep.values(), keys=Perc_rep.keys())
        res = [sub.split('m')[1] for sub in Perc_rep.columns]
        res = [r + " Months" for r in res]
        Perc_rep.columns = res
        Perc_rep = Perc_rep.dropna(axis = 1)

        for dista in Perc_rep.index.levels[0]:
            for tp in Perc_rep.columns:
                df_1 = dict(Perc_rep[tp][dista])
                keys = list(df_1.keys())
                keys.remove('True')
                vals = [df_1[x] for x in keys]
                idx = vals.index(min(np.array(vals) - 0.5)+0.5)
                if keys[idx] == df_1['True']:
                    Perf[dista][tp]['Correct'] = Perf[dista][tp]['Correct'] + 1
                else:
                    Perf[dista][tp]['Incorrect'] = Perf[dista][tp]['Incorrect'] + 1
                Perf[dista][tp]['Percent correct'] = Perf[dista][tp]['Correct']/(Perf[dista][tp]['Correct']+Perf[dista][tp]['Incorrect'])


Style_and_Save(copy.deepcopy(Perf), 'Long_Percent_Correct')

