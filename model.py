
import pandas as pd
import numpy as np
import pandas_profiling 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import math
plt.style.use('ggplot')
from matplotlib.font_manager import FontProperties
import matplotlib
chinese_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\kaiu.ttf')
import math
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 47)
pd.set_option('display.max_columns', 11)

# 資料讀取
dat1 = pd.read_csv('airline.csv')
dat2 = pd.read_csv('cache_map.csv')
dat3 = pd.read_csv('day_schedule.csv')
dat4 = pd.read_csv('group.csv')
dat5 = pd.read_csv('order.csv')
dat6 = pd.read_csv('training-set.csv')
dat7 = pd.read_csv('testing-set.csv')











