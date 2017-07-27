# InstaLocation_project
Master function, including necessary csvs, for finding locations in Instagram profile descriptions.

# Instructions:
First, run every cell/function after the cell that begins with "def everything" (there is a large bold note to mark the space in the ipython notebook). Then run the cell that begins with "def everything", and use it on a piece of text. This order must be followed to avoid errors!


# Necessary Files:
1) all_locations.csv : A csv of all of the locations used with partial word search. All of these locations greater than 3 letters. Includes countries, cities, nationalities/ethnicities, US state names, and flag emojis. Desired Column Name: "Name"

2) airport_info.csv: Contains a list of major airport names and their codes, to be made into a compiled regex for whole word search. Desired Column Name: "Code"

3) country_abbvs.csv: Contains a list of all the countries of the word along with their 2-letter codes (like AU for Australia). Will be made into a compiled regex for whole word search. Desired Column Name: "Abbreviation"

4) f2_lr_features.npy: File that contians the filter 2 features of our original groundtruth set, to be used as the features for the filter 2 models. 

5) groundtruth.csv : The "ground truth" of training data file. Can be read into a panda dataframe with the folowing columns: "matched_locations", "is_location?", "is_home?", "combined_descriptions", "original_location", "Context". The "is_location?" column contains the manually added response for filter 1 model.

6) ground_truth_features.npy: File that contains the filter 1 features of our original groundtruth_set, to be used as the features for the filter 1 models.

7) groundtruth_setup.csv: Contains the same columns as the above groundtruth.csv, except without the "is_location?" or "is_home?" columns, incase you wanted to redo the ground truth, or add more data to it. 

8)lr_filtered_for_f2.csv: A "ground_truth" file that is the result of using a Logistic Regression Model to predict actual locations, and creating a "filtered" list of only actual locations to be passed to filter 2. The "is_home?" column contains the manually added response for the filter 2 models. 
 
9) state_abbreviations.csv: Contains a list of the 50 US states and their 2-letter abbreviations. Used to make a compiled regex for whole word search. Desired Column Name: "Abbreviation"
