import pandas as pd
from pandas import ExcelWriter
import os

input_file = pd.read_excel('ff.xlsx')

def clean_dir(labels_list, K):
    """
    Removes the generated Excel files for each label and fold.
    """
    for label in labels_list:
        for i in range(1, K+1):
            os.remove(label + str(i) + ".xlsx")

def beginning():
    global input_file
    Category = "Category"
    Link = "Link"
    Label = "Label"
    K = 5

    labels_list = input_file[Label].unique()  # ['ETOH' 'DIA' 'VPA' 'CAF' 'UT']

    test_data = {}
    train_validate_data = {}

    for label in labels_list:
        # Split the data into test and train/validation sets for each label
        test_data[label] = input_file.loc[(input_file[Label] == label) & (input_file[Category] == 'TEST')]
        res = input_file.loc[(input_file[Label] == label) & ((input_file[Category] == 'VALIDATION') | (input_file[Category] == 'TRAIN'))].copy()
        res[Category] = 'TRAIN'

        set_size = round((res.shape[0] / K))
        t_size = res.shape[0]

        # Split the train/validation data into K folds for each label
        for i in range(1, K+1):
            start = (i - 1) * set_size
            stop = (i * set_size)
            if i == K:
                stop = t_size
            train_validate_data[(label + str(i))] = res.iloc[start:stop]

            # Save each fold as a separate Excel file
            writer = ExcelWriter(label + str(i) + '.xlsx')
            train_validate_data[(label + str(i))].to_excel(writer, index=False)
            writer.save()

    for i in range(1, K+1):
        data = []
        for label in labels_list:
            # Read the data for each label and fold from the generated Excel files
            f = pd.read_excel(label + str(i) + '.xlsx')
            res = f.loc[f[Label] == label]
            res[Category] = "VALIDATION"
            data.append(res)
            data.append(test_data[label])

            # Combine the data from other folds for each label
            for marker in range(1, K+1):
                if marker != i:
                    f = pd.read_excel(label + str(marker) + '.xlsx')
                    data.append(f.loc[f[Label] == label])

        # Concatenate all the data into a single DataFrame
        dfObj = pd.concat(data, sort=False, ignore_index=True)

        # Save the combined data as a new Excel file for each fold
        writer = ExcelWriter('output_data' + str(i) + '.xlsx')
        dfObj.to_excel(writer, sheet_name='OUTPUT', index=False)
        writer.save()

    # Clean up the generated Excel files
    clean_dir(labels_list, K)