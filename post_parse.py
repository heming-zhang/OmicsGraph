import os
import re
import numpy as np
import pandas as pd

from load_data import LoadData

# Randomize the survival label
def input_random(randomized, graph_output_folder):
    randomized = False
    if randomized == True:
        random_survival_filtered_feature_df = survival_filtered_feature_df.sample(frac = 1).reset_index(drop=True)
        random_survival_filtered_feature_df.to_csv(os.path.join(graph_output_folder, 'random-survival-label.csv'), index=False)
    else:
        random_survival_filtered_feature_df = pd.read_csv(os.path.join(graph_output_folder, 'random-survival-label.csv'))
    print(random_survival_filtered_feature_df)

# Split deep learning input into training and test
def split_k_fold(k, graph_output_folder):
    random_survival_filtered_feature_df = pd.read_csv(os.path.join(graph_output_folder, 'random-survival-label.csv'))
    num_points = random_survival_filtered_feature_df.shape[0]
    num_div = int(num_points / k)
    num_div_list = [i * num_div for i in range(0, k)]
    num_div_list.append(num_points)
    # Split [random_survival_filtered_feature_df] into [k] folds
    for place_num in range(k):
        low_idx = num_div_list[place_num]
        high_idx = num_div_list[place_num + 1]
        print('\n--------TRAIN-TEST SPLIT WITH TEST FROM ' + str(low_idx) + ' TO ' + str(high_idx) + '--------')
        split_input_df = random_survival_filtered_feature_df[low_idx : high_idx]
        split_input_df.to_csv(os.path.join(graph_output_folder, 'split-random-survival-label-' + str(place_num + 1) + '.csv'), index=False)
        print(split_input_df.shape)


# ############## MOUDLE 1 ################
# ### Randomize the survival label
# randomized = False
# graph_output_folder = 'graph-data'
# input_random(randomized=randomized, graph_output_folder=graph_output_folder)
# ### Split deep learning input into 5-fold
# split_k_fold(k=5, graph_output_folder=graph_output_folder)
# ### Load all split data into graph format
# processed_dataset = 'UCSC-process'
# if os.path.exists('./' +graph_output_folder + '/form_data') == False:
#     os.mkdir('./' +graph_output_folder + '/form_data')
# k = 5
# batch_size = 64
# LoadData().load_all_split(batch_size, k, processed_dataset, graph_output_folder)

# # ############## MOUDLE 2 ################
# graph_output_folder = 'graph-data'
# processed_dataset = 'UCSC-process'
# LoadData().load_adj_edgeindex(graph_output_folder)

################ MOUDLE 3 ################
# # FORM N-TH FOLD TRAINING DATASET
# k = 5
# n_fold = 1
# graph_output_folder = 'graph-data'
# LoadData().load_train_test(k, n_fold, graph_output_folder)

# Check the default ratio of survival
n_fold = 1
graph_output_folder = 'graph-data'
form_data_path = './' + graph_output_folder + '/form_data'
yTr =  np.load(form_data_path + '/yTr' + str(n_fold) + '.npy')
num_elements = yTr.size
num_ones = np.count_nonzero(yTr) # Dead is one
num_zeros = num_elements - num_ones # Alive is zero
ratio_of_ones = num_ones / num_elements
ratio_of_zeros = num_zeros / num_elements
print("Ratio of Dead:", ratio_of_ones)
print("Ratio of Alive:", ratio_of_zeros)