import numpy as np
import pandas as pd

import abc
import logging
import os
import zipfile
from typing import Tuple
from urllib.request import urlopen

uci_datasets = ['boston',
 'concrete',
 'protein',
 'bike',
 'star', 
 'concrete', 
 'bio']





def load_seer_cutract_dataset(name="seer", seed=42, imbalance=False):
    """
    The function `load_seer_cutract_dataset` loads and preprocesses a dataset for prostate cancer
    
    Args:
      name: The `name` parameter is used to specify the name of the dataset to load. The default value
    is "seer". Defaults to seer
      seed: The `seed` parameter
      imbalance: The `imbalance` parameter is a boolean flag that determines whether to create an
    imbalanced dataset or not. 
    
    Returns:
      The function `load_seer_cutract_dataset` returns three values: `df[features]`, `df[label]`, and
    `df`.
    """
    # third party
    import pandas as pd
    import sklearn

    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5


    # Features to keep
    features = [
        "age",
        "mortCancer",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
    ]

    label = "psa"
    df = pd.read_csv(f"../data/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df["mortCancer"] == True  # noqa: E712
    df_dead = df[mask]
    df_survive = df[~mask]

    if imbalance==True:
        if name == "seer":
            df = df.sample(10000, random_state=seed)
        else:
            df = df.sample(2000, random_state=seed)

        df = sklearn.utils.shuffle(df, random_state=seed)
        df = df.reset_index(drop=True)
        return df[features], df[label], df



    if name == "seer":
        n_samples = 10000
        ns = 10000
    else:
        n_samples = 1000
        ns = 1000
    df = pd.concat(
        [
            df_dead.sample(ns, random_state=seed),
            df_survive.sample(n_samples, random_state=seed),
        ],
    )

    df = sklearn.utils.shuffle(df, random_state=seed)
    df = df.reset_index(drop=True)
    return df[features], df[label], df



_ALL_REGRESSION_DATASETS = {}


def add_regression(C):
    _ALL_REGRESSION_DATASETS.update({C.name: C})
    return C


# The Dataset class is a template for creating objects that represent a collection of data.
class Dataset:
    def __init__(self, name: str, url: str, directory: str):
        self.name = name
        self.url = url
        self.directory = directory

    @property
    def datadir(self):
        return os.path.join(self.directory, self.name)

    @property
    def datapath(self):
        return os.path.join(self.datadir, self.url.split("/")[-1])

    @abc.abstractmethod
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def shuffle(self, X: np.array, Y: np.array, seed: int = 0):
        """
        The function shuffles the rows of two numpy arrays, X and Y, using a given seed value.
        
        Returns:
          two arrays, X[perm] and Y[perm], which are the shuffled versions of the input arrays X and Y.
        """
        N = X.shape[0]
        perm = np.arange(N)
        np.random.seed(seed)
        np.random.shuffle(perm)
        return X[perm], Y[perm]

    def normalize(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The normalize function takes an input array Z, calculates the mean and standard deviation along
        the 0th axis, and returns the normalized array, mean, and standard deviation.
        
        Args:
          Z (np.ndarray): Z is a numpy array containing the data that needs to be normalized.
        
        Returns:
          a tuple containing three numpy arrays: the normalized array (Z - Z_mean) / Z_std, the mean
        array Z_mean, and the standard deviation array Z_std.
        """
        Z_mean = np.mean(Z, 0, keepdims=True)
        Z_std = 1e-6 + np.std(Z, 0, keepdims=True)
        return (Z - Z_mean) / Z_std, Z_mean, Z_std

    def preprocess(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The `preprocess` function normalizes the input and output arrays and returns them along with the
        mean and standard deviation values.
        
        Args:
          X (np.ndarray): The parameter X is an input array of shape (n_samples, n_features) containing
        the input data for preprocessing.
          Y (np.ndarray): The parameter Y is a numpy array containing the target values or labels for
        the given input data X.
        
        Returns:
          The preprocess method returns two numpy arrays, X and Y, after applying normalization to them.
        """
        X, self.X_mean, self.X_std = self.normalize(X)
        Y, self.Y_mean, self.Y_std = self.normalize(Y)
        return X, Y

    def split(
            self,
            X: np.array,
            Y: np.array,
            prop_train: float = 0.8,
            prop_val: float = 0.1,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        
        """
        The `split` function takes in input data `X` and target labels `Y`, and splits them into
        training, validation, and test sets based on the specified proportions.
        
 
        Returns:
          three tuples. The first tuple contains two numpy arrays, which are the training data X and Y.
        The second tuple contains two numpy arrays, which are the validation data X and Y. The third
        tuple contains two numpy arrays, which are the test data X and Y.
        """

        N = X.shape[0]
        n_train = int(N * prop_train)
        n_val = int(N * prop_val)
        train_data = X[:n_train], Y[:n_train]
        val_data = X[n_train: n_train + n_val], Y[n_train: n_train + n_val]
        test_data = X[n_train + n_val:], Y[n_train + n_val:]
        return train_data, val_data, test_data

  
    def load(
            self,
            prop_train: float = 0.8,
            prop_val: float = 0.1,
            batch_size: int = 128,
            shuffle_train: bool = False,
            prefetch: bool = True,
            seed: bool = False,
    ):
        
        """
        The `load` function takes in several parameters, reads data from a source, and returns the data
        along with an optional seed value.
        
        Returns:
          the variables X and Y. If the seed parameter is not False, it will also return the seed value.
        """
        
        X, Y = self.read()
        if seed!=False:
            return X, Y, seed
        else:
            return X,Y

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        """
        The download function is used to download files.
        """
        if self.needs_download:
            logging.info("\nDownloading {} data...".format(self.name))

            if not os.path.isdir(self.datadir):
                os.mkdir(self.datadir)

            filename = os.path.join(self.datadir, self.url.split("/")[-1])
            with urlopen(self.url) as response, open(filename, "wb") as out_file:
                data = response.read()
                out_file.write(data)

            is_zipped = np.any([z in self.url for z in [".gz", ".zip", ".tar"]])
            if is_zipped:
                zip_ref = zipfile.ZipFile(filename, "r")
                zip_ref.extractall(self.datadir)
                zip_ref.close()

            logging.info("Download completed.".format(self.name))
        else:
            logging.info("{} dataset is already available.".format(self.name))


uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
datamap = {"boston": 100,
           "concrete": 1000,
           "protein": 100,
           
           }
  

@add_regression
class Boston(Dataset):
    name = "boston"
    url = uci_base_url + "housing/housing.data"

    def __init__(self, directory, name=name, url=url):
        super().__init__(name=name, url=url, directory=directory)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_fwf(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Concrete(Dataset):
    name = "concrete"
    url = uci_base_url + "concrete/compressive/Concrete_Data.xls"

    def __init__(self, directory, name=name, url=url):
        super().__init__(name=name, url=url, directory=directory)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)
    
@add_regression
class Protein(Dataset):
    name = "protein"
    url = uci_base_url + "00265/CASP.csv"

    def __init__(self, directory, name=name, url=url):
        super().__init__(name=name, url=url, directory=directory)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(self.datapath).values
        return data[:, 1:], data[:, 0].reshape(-1, 1)




ALL_REGRESSION_DATASETS = _ALL_REGRESSION_DATASETS

regression_datasets = list(_ALL_REGRESSION_DATASETS.keys())
regression_datasets.sort()


def download_regression_dataset(name, directory, *args, **kwargs):
    dataset = _ALL_REGRESSION_DATASETS[name](directory, *args, **kwargs)
    dataset.download()


def download_all_regression_datasets(directory, *args, **kwargs):
    for name in list(_ALL_REGRESSION_DATASETS.keys()):
        download_regression_dataset(name, directory, *args, **kwargs)
        
def load_regression_dataset(
    name, dir, seeded=False, *args, **kwargs
):
    dataset = _ALL_REGRESSION_DATASETS[name](dir)
    if seeded==True:
        seed = datamap[name]
        return dataset.load(seed=seed, *args, **kwargs)
    else:   
        return dataset.load(*args, **kwargs)


mapper = {}
for i, val in enumerate(regression_datasets):
    mapper[val] = i



def GetDataset(name, base_path, seeded=False):
    """ Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	"""
    if name=="star":
        df = pd.read_csv(base_path + 'STAR.csv')
        df.loc[df['gender'] == 'female', 'gender'] = 0
        df.loc[df['gender'] == 'male', 'gender'] = 1
        
        df.loc[df['ethnicity'] == 'cauc', 'ethnicity'] = 0
        df.loc[df['ethnicity'] == 'afam', 'ethnicity'] = 1
        df.loc[df['ethnicity'] == 'asian', 'ethnicity'] = 2
        df.loc[df['ethnicity'] == 'hispanic', 'ethnicity'] = 3
        df.loc[df['ethnicity'] == 'amindian', 'ethnicity'] = 4
        df.loc[df['ethnicity'] == 'other', 'ethnicity'] = 5
        
        df.loc[df['stark'] == 'regular', 'stark'] = 0
        df.loc[df['stark'] == 'small', 'stark'] = 1
        df.loc[df['stark'] == 'regular+aide', 'stark'] = 2
        
        df.loc[df['star1'] == 'regular', 'star1'] = 0
        df.loc[df['star1'] == 'small', 'star1'] = 1
        df.loc[df['star1'] == 'regular+aide', 'star1'] = 2        
        
        df.loc[df['star2'] == 'regular', 'star2'] = 0
        df.loc[df['star2'] == 'small', 'star2'] = 1
        df.loc[df['star2'] == 'regular+aide', 'star2'] = 2   

        df.loc[df['star3'] == 'regular', 'star3'] = 0
        df.loc[df['star3'] == 'small', 'star3'] = 1
        df.loc[df['star3'] == 'regular+aide', 'star3'] = 2      
        
        df.loc[df['lunchk'] == 'free', 'lunchk'] = 0
        df.loc[df['lunchk'] == 'non-free', 'lunchk'] = 1
        
        df.loc[df['lunch1'] == 'free', 'lunch1'] = 0    
        df.loc[df['lunch1'] == 'non-free', 'lunch1'] = 1      
        
        df.loc[df['lunch2'] == 'free', 'lunch2'] = 0    
        df.loc[df['lunch2'] == 'non-free', 'lunch2'] = 1  
        
        df.loc[df['lunch3'] == 'free', 'lunch3'] = 0    
        df.loc[df['lunch3'] == 'non-free', 'lunch3'] = 1  
        
        df.loc[df['schoolk'] == 'inner-city', 'schoolk'] = 0
        df.loc[df['schoolk'] == 'suburban', 'schoolk'] = 1
        df.loc[df['schoolk'] == 'rural', 'schoolk'] = 2  
        df.loc[df['schoolk'] == 'urban', 'schoolk'] = 3

        df.loc[df['school1'] == 'inner-city', 'school1'] = 0
        df.loc[df['school1'] == 'suburban', 'school1'] = 1
        df.loc[df['school1'] == 'rural', 'school1'] = 2  
        df.loc[df['school1'] == 'urban', 'school1'] = 3      
        
        df.loc[df['school2'] == 'inner-city', 'school2'] = 0
        df.loc[df['school2'] == 'suburban', 'school2'] = 1
        df.loc[df['school2'] == 'rural', 'school2'] = 2  
        df.loc[df['school2'] == 'urban', 'school2'] = 3      
        
        df.loc[df['school3'] == 'inner-city', 'school3'] = 0
        df.loc[df['school3'] == 'suburban', 'school3'] = 1
        df.loc[df['school3'] == 'rural', 'school3'] = 2  
        df.loc[df['school3'] == 'urban', 'school3'] = 3  
        
        df.loc[df['degreek'] == 'bachelor', 'degreek'] = 0
        df.loc[df['degreek'] == 'master', 'degreek'] = 1
        df.loc[df['degreek'] == 'specialist', 'degreek'] = 2  
        df.loc[df['degreek'] == 'master+', 'degreek'] = 3 

        df.loc[df['degree1'] == 'bachelor', 'degree1'] = 0
        df.loc[df['degree1'] == 'master', 'degree1'] = 1
        df.loc[df['degree1'] == 'specialist', 'degree1'] = 2  
        df.loc[df['degree1'] == 'phd', 'degree1'] = 3              
        
        df.loc[df['degree2'] == 'bachelor', 'degree2'] = 0
        df.loc[df['degree2'] == 'master', 'degree2'] = 1
        df.loc[df['degree2'] == 'specialist', 'degree2'] = 2  
        df.loc[df['degree2'] == 'phd', 'degree2'] = 3
        
        df.loc[df['degree3'] == 'bachelor', 'degree3'] = 0
        df.loc[df['degree3'] == 'master', 'degree3'] = 1
        df.loc[df['degree3'] == 'specialist', 'degree3'] = 2  
        df.loc[df['degree3'] == 'phd', 'degree3'] = 3          
        
        df.loc[df['ladderk'] == 'level1', 'ladderk'] = 0
        df.loc[df['ladderk'] == 'level2', 'ladderk'] = 1
        df.loc[df['ladderk'] == 'level3', 'ladderk'] = 2  
        df.loc[df['ladderk'] == 'apprentice', 'ladderk'] = 3  
        df.loc[df['ladderk'] == 'probation', 'ladderk'] = 4
        df.loc[df['ladderk'] == 'pending', 'ladderk'] = 5
        df.loc[df['ladderk'] == 'notladder', 'ladderk'] = 6
        
        
        df.loc[df['ladder1'] == 'level1', 'ladder1'] = 0
        df.loc[df['ladder1'] == 'level2', 'ladder1'] = 1
        df.loc[df['ladder1'] == 'level3', 'ladder1'] = 2  
        df.loc[df['ladder1'] == 'apprentice', 'ladder1'] = 3  
        df.loc[df['ladder1'] == 'probation', 'ladder1'] = 4
        df.loc[df['ladder1'] == 'noladder', 'ladder1'] = 5
        df.loc[df['ladder1'] == 'notladder', 'ladder1'] = 6
        
        df.loc[df['ladder2'] == 'level1', 'ladder2'] = 0
        df.loc[df['ladder2'] == 'level2', 'ladder2'] = 1
        df.loc[df['ladder2'] == 'level3', 'ladder2'] = 2  
        df.loc[df['ladder2'] == 'apprentice', 'ladder2'] = 3  
        df.loc[df['ladder2'] == 'probation', 'ladder2'] = 4
        df.loc[df['ladder2'] == 'noladder', 'ladder2'] = 5
        df.loc[df['ladder2'] == 'notladder', 'ladder2'] = 6
        
        df.loc[df['ladder3'] == 'level1', 'ladder3'] = 0
        df.loc[df['ladder3'] == 'level2', 'ladder3'] = 1
        df.loc[df['ladder3'] == 'level3', 'ladder3'] = 2  
        df.loc[df['ladder3'] == 'apprentice', 'ladder3'] = 3  
        df.loc[df['ladder3'] == 'probation', 'ladder3'] = 4
        df.loc[df['ladder3'] == 'noladder', 'ladder3'] = 5
        df.loc[df['ladder3'] == 'notladder', 'ladder3'] = 6
        
        df.loc[df['tethnicityk'] == 'cauc', 'tethnicityk'] = 0
        df.loc[df['tethnicityk'] == 'afam', 'tethnicityk'] = 1
        
        df.loc[df['tethnicity1'] == 'cauc', 'tethnicity1'] = 0
        df.loc[df['tethnicity1'] == 'afam', 'tethnicity1'] = 1
        
        df.loc[df['tethnicity2'] == 'cauc', 'tethnicity2'] = 0
        df.loc[df['tethnicity2'] == 'afam', 'tethnicity2'] = 1
        
        df.loc[df['tethnicity3'] == 'cauc', 'tethnicity3'] = 0
        df.loc[df['tethnicity3'] == 'afam', 'tethnicity3'] = 1
        df.loc[df['tethnicity3'] == 'asian', 'tethnicity3'] = 2
        
        df = df.dropna()
        
        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]
        
        
        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[0:8],names[17:]))
        X = df.loc[:, data_names].values
        y = grade.values
        seed=10
        
        
    if name=="bio":
        #https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(base_path + 'CASP.csv')        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values   
        seed=100
   
        
    
    if name == "concrete":
        dataset = np.loadtxt(open(base_path + 'Concrete_Data.csv', "rb"), delimiter=",", skiprows=1)
        X = dataset[:, :-1]
        y = dataset[:, -1:]
        seed=1000

    
    if name=="bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df=pd.read_csv(base_path + 'bike_train.csv')
        
        # # seperating season as per values. this is bcoz this will enhance features.
        season=pd.get_dummies(df['season'],prefix='season')
        df=pd.concat([df,season],axis=1)
        
        # # # same for weather. this is bcoz this will enhance features.
        weather=pd.get_dummies(df['weather'],prefix='weather')
        df=pd.concat([df,weather],axis=1)
        
        # # # now can drop weather and season.
        df.drop(['season','weather'],inplace=True,axis=1)
        df.head()
        
        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011:0, 2012:1})
 
        df.drop('datetime',axis=1,inplace=True)
        df.drop(['casual','registered'],axis=1,inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count',axis=1).values
        y = df['count'].values
        seed=10

        
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    if seeded!=False:
        return X, y, seed
    else:
        
        return X, y, 


def load_medical(dataset, seeded=False):
    """
    The function "load_medical" loads a given medical dataset.
    
    Args:
      dataset: The dataset parameter is the name or path of the medical dataset that you want to load.
      seeded: boolean 
    """
    if dataset == "cf":

        # outcome_variable = 'FEV1'
        outcome_variable = "time_to_event"
        df_static = pd.read_csv("../data/cf_static_train_data.csv")
        df_temporal = pd.read_csv("../data/cf_temporal_train_data_eav.csv")
        processing_needed = True
        seed=10

    if dataset == "ward":

        outcome_variable = "Glasgow Coma Scale Score"
        outcome_variable = "WHITE BLOOD CELL COUNT"
        df_static = pd.read_csv("../data/ward_static_train_data.csv")
        df_temporal = pd.read_csv("../data/ward_temporal_train_data_eav.csv")
        processing_needed = True
        seed=10

    if dataset == "mimic_antibiotics":
        outcome_variable = "wbc"
        df_static = pd.read_csv("../data/mimic_antibiotics_static_train_data.csv")
        df_temporal = pd.read_csv("../data/mimic_antibiotics_temporal_train_data_eav.csv")
        processing_needed = True
        seed=100

    if dataset == "los":
        outcome_variable = "lengthofstay"
        df = pd.read_csv("../data/LengthOfStay.csv")
        dropcols = ["eid", "vdate", "discharged", "facid"]
        df = df.drop(columns=dropcols)

        from sklearn import preprocessing

        le1 = preprocessing.LabelEncoder()
        df["rcount"] = le1.fit_transform(df["rcount"])
        le2 = preprocessing.LabelEncoder()
        df["gender"] = le2.fit_transform(df["gender"])
        processing_needed = False
        seed=10

    if seeded!=False:
        if processing_needed:
            return df_static, df_temporal, outcome_variable, processing_needed, seed
        else:
            return df, outcome_variable, processing_needed, seed
    
    else:
        if processing_needed:
            return df_static, df_temporal, outcome_variable, processing_needed
        else:
            return df, outcome_variable, processing_needed
        

# The TrainData class is a custom dataset class that takes in X_data and y_data and allows for
# indexing and length retrieval.
class TrainData(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)


