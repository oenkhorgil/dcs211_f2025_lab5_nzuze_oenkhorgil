import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 

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

    data = df.to_numpy()
    data = np.nan_to_num(data)  
    return data

#########################

def predictiveModel(train_data, test_features):

    # Separate the features (X) and labels (y)
    X_train = train_data[:, :-1]       # pixel columns
    y_train = train_data[:, -1]        # label column

    # get Euclidean distance between this test example and all training rows
    distances = np.linalg.norm(X_train - test_features, axis=1)

    # Find the index of the smallest distance (the nearest neighbor)
    nearest_index = np.argmin(distances)

    # Return the label of that nearest neighbor
    predicted_label = int(y_train[nearest_index])
    return predicted_label

###################



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
    # OK!  Onward to knn for digits! (based on your iris work...)
    #

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()
