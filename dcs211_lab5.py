import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###########################################################################
def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

###########################################################################
def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple


###################
def cleanTheData(df: pd.DataFrame) -> np.ndarray:

    df_clean = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df_clean = df_clean.dropna(axis=1, how="all")

    X = df_clean.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df_clean.iloc[:, -1].apply(pd.to_numeric, errors="coerce").fillna(0)

    X = X.to_numpy(dtype=int)
    y = y.to_numpy(dtype=int).reshape(-1, 1)

    data = np.hstack([X, y])

    return data

#########################
def predictiveModel(train_data: np.ndarray, test_features: np.ndarray) -> int:

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].ravel()

    diffs = X_train - test_features
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    i_min = int(np.argmin(dists))
    return int(y_train[i_min])

###################
def compareLabels(y_true: np.ndarray, y_pred: np.ndarray, title: str="Results") -> str:
    ''' Compares true and predicted labels and returns a summary.
    Parameters:
        y_true: 1D numpy array or list of actual labels.
        y_pred: 1D numpy array or list of predicted labels.
        title: optional string used as a heading for the output (default is "Results").
    Returns:
        a formatted string containing the accuracy, confusion matrix, 
        and classification report.
    '''
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=3)
    return (
        f"{title}\n"
        f"Accuracy: {acc:.3f}\n"
        f"Confusion matrix (rows=true, cols=pred):\n{cm}\n"
        f"Classification report:\n{rep}\n"
    )

##############
def pick_initial_k(n_train: int) -> int:
    ''' Selects an initial value for k in k-NN based on the training set size.
    Parameters:
        n_train: integer representing the number of training samples.
    Returns:
        an odd integer k approximately equal to √n_train, 
        limited to the range [1, 31] to avoid extreme values.
    '''
    k0 = int(round(np.sqrt(max(1, n_train))))
    if k0 % 2 == 0:
        k0 += 1
    return max(1, min(k0, 31))

##############
def _knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    ''' Uses KNeighborsClassifier to make predictions.
    Parameters:
        X_train: 2D numpy array of training features.
        y_train: 1D numpy array of corresponding training labels.
        X_test: 2D numpy array of test features.
        k: integer specifying the number of nearest neighbors.
    Returns:
        a 1D numpy array of predicted labels for the test data.
    '''
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

##############

def splitData(all_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the full [features..., label] array into test and train parts.
    Returns [X_test, y_test, X_train, y_train] in that order.
    Uses NumPy's *global* RNG state (seeded externally) via np.random.permutation.
    Test size is 20% of samples (at least 1 and at most n-1).
    """
    arr = np.asarray(all_data)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("all_data must be 2D with at least 2 columns")

    X = arr[:, :-1]
    y = arr[:, -1].astype(int)

    n = len(y)
    n_test = max(1, min(n - 1, int(np.floor(0.2 * n))))
    perm = np.random.permutation(n)

    test_idx  = perm[:n_test]
    train_idx = perm[n_test:]

    X_test,  y_test  = X[test_idx],  y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    return [X_test, y_test, X_train, y_train]

##############
def findBestK(X_train: np.ndarray, y_train: np.ndarray, k_values:list[int]=None, validation_size: float = 0.2, seed=None) -> Tuple[int, dict[int, float]]:
    ''' Finds the best k value for k-NN using a validation split.
    Parameters:
        X_train: 2D numpy array of training features.
        y_train: 1D numpy array of corresponding training labels.
        k_values: optional list of candidate k values to test (default is odd k from 1 to 31).
        validation_size: fraction of data used for validation (default 0.2).
        seed: optional integer for reproducible random splits.
    Returns:
        a tuple (best_k, scores) where:
            best_k is the k value with the highest validation accuracy
            scores is a dictionary mapping each k to its validation accuracy.
    '''
    if k_values is None:
        k_values = [k for k in range(1, 32) if k % 2 == 1]

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=seed, stratify=y_train
    )

    scores = {}
    for k in k_values:
        preds = _knn_predict(X_sub, y_sub, X_val, k)
        scores[k] = float(np.mean(preds == y_val))

    max_acc = max(scores.values())
    best_k = min(k for k, s in scores.items() if s == max_acc)
    return best_k, scores

##############
def trainAndTest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, best_k: int) -> np.ndarray:
    ''' Trains a k-NN model on the full training set and predicts labels for the test set.
    Parameters:
        X_train: 2D numpy array of training features.
        y_train: 1D numpy array of corresponding training labels.
        X_test: 2D numpy array of test features.
        best_k: integer specifying the chosen number of nearest neighbors.
    Returns:
        a 1D numpy array of predicted labels for the test data.
    '''
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    clf = KNeighborsClassifier(n_neighbors=int(best_k))
    clf.fit(X_train, np.asarray(y_train).ravel())
    preds = clf.predict(X_test)
    return preds.astype(int)



##############
def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")

    data= cleanTheData(df)
    n = len(data)
    split_idx = int(0.8 * n)

    train_data = data[:split_idx]
    test_data  = data[split_idx:]
##
    y_test = test_data[:, -1].astype(int)
    predicted_labels = []
    for row in tqdm(test_data, desc="Manual 1-NN (train first 80%)"):
        test_features = row[:-1]                 
        prediction = predictiveModel(train_data, test_features)
        predicted_labels.append(prediction)
    predicted_labels = np.array(predicted_labels, dtype=int)
 
    accuracy = np.round(np.mean(predicted_labels == y_test), 3)
    print(f"Accuracy: {accuracy:.3f}")
##
    train_data_swapped = data[n - split_idx :]   # last 80%
    test_data_swapped  = data[: n - split_idx]   # first 20%
    y_test_swapped = test_data_swapped[:, -1].astype(int)
    
    predicted_labels_swapped = []
    for row in tqdm(test_data_swapped, desc="Manual 1-NN (train last 80%)"):
        test_features = row[:-1]
        prediction = predictiveModel(train_data_swapped, test_features)
        predicted_labels_swapped.append(prediction)

    predicted_labels_swapped = np.array(predicted_labels_swapped, dtype=int)

    accuracy_swapped = np.round(np.mean(predicted_labels_swapped == y_test_swapped), 3)
    print(f"Accuracy train: {accuracy_swapped:.3f}")
##
    num_to_draw = 5
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)

        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()

    #
    label_col = df.columns[0]       
    feature_cols = df.columns[1:]
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[label_col].to_numpy()
    X_test, y_test, X_train, y_train = splitData(
    X_all=X_all, y_all=y_all, test_size=0.2, random_state=8675309)
    #
    k_guess = pick_initial_k(len(y_train))
    y_pred_guess = trainAndTest(X_train, y_train, X_test, k_guess)
    print(compareLabels(y_test, y_pred_guess, title=f"Exercise 8 – k guess = {k_guess}"))

    #
    seeds = [8675309, 5551212, 20251109]
    best_by_seed = {}
    score_tables = {}

    for s in seeds:
        k_s, scores_s = findBestK(X_train, y_train, seed=s)
        best_by_seed[s] = k_s
        score_tables[s] = scores_s

    best_k = int(np.median(list(best_by_seed.values()))) 
    print("Exercise 9 – Best k per seed:", best_by_seed)
    print("Exercise 9 – Chosen best_k:", best_k)

    #
    y_pred_best = trainAndTest(X_train, y_train, X_test, best_k)
    print(compareLabels(y_test, y_pred_best, title=f"Exercises 9–10 – best_k = {best_k}"))
    
    if __name__ == "__main__":
        main()
