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
    plt.show(block = True)

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
    """
    Accepts a pandas DataFrame and returns a NumPy array
    of the cleaned data).
    """
    # keep only pix0..pix63 and the actual_digit label
    pix_cols = [f"pix{i}" for i in range(64)]
    df_clean = df[pix_cols + ["actual_digit"]]

    # return as numpy array
    return df_clean.to_numpy()

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
    # Step 1: Read the file
    filename = 'digits.csv'
    df = pd.read_csv(filename, header=0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")

    # Step 2: Clean the data
    data = cleanTheData(df)
    n = len(data)
    split_idx = int(0.8 * n)

    # Step 3: Train = first 80%, Test = last 20%
    train_data = data[:split_idx]
    test_data  = data[split_idx:]
    y_test = test_data[:, -1].astype(int)

    predicted_labels = []
    for row in tqdm(test_data, desc="Manual 1-NN (train first 80%)"):
        test_features = row[:-1]
        prediction = predictiveModel(train_data, test_features)
        predicted_labels.append(prediction)

    predicted_labels = np.array(predicted_labels, dtype=int)
    accuracy = (predicted_labels == y_test).mean()
    print(f"Accuracy 1st train: {accuracy:.3f}")

    # Step 4: Swap train/test for Q4
    train_data_swapped = data[n - split_idx:]
    test_data_swapped  = data[:n - split_idx]
    y_test_swapped = test_data_swapped[:, -1].astype(int)

    predicted_labels_swapped = []
    for row in tqdm(test_data_swapped, desc="Manual 1-NN (train last 80%)"):
        test_features = row[:-1]
        prediction = predictiveModel(train_data_swapped, test_features)
        predicted_labels_swapped.append(prediction)

    predicted_labels_swapped = np.array(predicted_labels_swapped, dtype=int)
    accuracy_swapped = (predicted_labels_swapped == y_test_swapped).mean()
    print(f"Accuracy train swapped: {accuracy_swapped:.3f}")

    # Step 5: Visualize the first five misclassified digits (Q5)
    wrong_idx = np.where(predicted_labels != y_test)[0]

    if len(wrong_idx) == 0:
        print("No misclassifications in split 1. Nothing to visualize.")
    else:
        print("Saving first up to five misclassifications from split 1:")
    for k in wrong_idx[:5]:
        abs_idx = split_idx + k
        digit, pixels = fetchDigit(df, abs_idx)
        print(f"Pred {predicted_labels[k]} vs True {y_test[k]} at abs row {abs_idx}")

        
        drawDigitHeatmap(pixels, showNumbers=True)

        # save to file
        fname = f"misclassified_{k}_pred{predicted_labels[k]}_true{y_test[k]}_abs{abs_idx}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  -> saved {fname}")



# Keep this at the bottom so it only runs when executed directly
if __name__ == "__main__":
    main()
