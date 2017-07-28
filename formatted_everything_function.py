
# coding: utf-8

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mpl


# **Note: Instructions on how we generated the training data is at the bottom of the notebook**




#files
ground_truth = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Ground_truths/groundtruth.csv")
f2_lr_filtered = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/lr_filtered_for_f2.csv")
all_partial_locs = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Location_info/all_locations.csv")
#regexs for whole word search
airports = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Location_info/airport_info.csv")
airport_reg = re.compile(r"\b({})\b".format("|".join(airports['Code'])))
country_abvs= pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Location_info/country_abbvs.csv")
country_abv_reg = re.compile(r"\b({})\b".format('|'.join(str(v) for v in country_abvs["Abbreviation"])))
upper_state_abv = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Location_info/state_abbreviations.csv")
upper_state_abv_reg = re.compile(r"\b({})\b".format("|".join(upper_state_abv["Abbreviation"])))
#set for a quick search for a feature
ethnicities = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Location_info/nationalities.csv")
set_ethnicities = set(ethnicities['Nationality'])


# **Note** If you want the ground_truth file before we added our manual 1's and 0's, here it is:
set_up_groundtruth = pd.read_csv("/Users/frances/Documents/Tribe/ops/InstaLocation_project/Ground_truths/groundtruth_setup.csv")

# # Important: Run all little functions before big one

# For ease of reading, and clarity, I put the big main function first, but all the little helper functions must be defined before you run the big one so the computer knows what the variables are referencing

def everything(text):
    """The function that consolidates the whole location process in one function.
    text: The IG profile description text, as a string.
    Returns a numpy array of only 'home' locations- that is, only the loctions in which the
    model was more than 50% confident that it was an actual location and more than 50% confident
    that it's a home location."""
    #Step 1: Put text into correct format
    text_df = pd.DataFrame(data = np.array([text]),
                           columns = np.array(["combined_descriptions"]))

    #Step 2: Extract Locations
    with_locations_df = extract_location(text_df, "combined_descriptions")

    #Step 3: Expand Dataframe - one location per row
    expanded = expand_location_df(with_locations_df)

    #Step 4: Find the Original Location
    with_original = make_origin_loc_column(expanded, "combined_descriptions", "matched_location")

    #Step 5: Extract the whole context
    with_context = make_context_column(with_original, "combined_descriptions", "original_location")

    #Step 6: Extract Before and After Context
    everything_df = make_before_and_after_columns(with_context, "Context", "original_location")

    #Step 7: Calculate Filter 1 Features
    feature_list = return_features_list(everything_df)
    cv_word_features = f1_make_cv_word_features(everything_df["Context"])
    f1_df_features_lst = df_features(everything_df)
    f1_df_features = return_df_features(f1_df_features_lst, everything_df) #into proper format
    all_features = combine_feature_lists(feature_list, cv_word_features, f1_df_features)
    top_50 = all_features[:,f1_inx]
    f1_features = np.array(list(top_50)) #formatting

    #Step 8: Feed to Filter 1 Model
    lr_prediction = f1_lr.predict_proba(f1_features)
    lr_loc_probs = [prob[1] for prob in lr_prediction]
    everything_df["f1_prob"] = lr_loc_probs
    everything_df["f1_binary"] = (everything_df["f1_prob"] > 0.5).astype(int)

    #Step 9: Filter Out Non-Locations
    actual_locations = everything_df.where(everything_df["f1_prob"] > 0.50).dropna()

    #Step 10: Calculate Filter 2 Features
    f2_feature_list = f2_return_features_list(actual_locations)
    f2_cv_word_features = f2_make_cv_word_features(actual_locations["Context"])
    f2_df_features_lst = df_features(actual_locations)
    f2_df_features = return_df_features(f2_df_features_lst, actual_locations)
    f2_all_features = combine_feature_lists(f2_feature_list, f2_cv_word_features, f2_df_features)
    f2_top_50 = f2_all_features[:,f2_inx]
    f2_features = np.array(list(f2_top_50)) #formatting

    #Step 11: Feed to Filter 2 Model
    f2_lr_prediction = f2_lr.predict_proba(f2_features)
    f2_lr_loc_probs = [prob[1] for prob in f2_lr_prediction]
    actual_locations["f2_prob"] = f2_lr_loc_probs
    actual_locations["f2_binary"] = (actual_locations["f1_prob"] > 0.5).astype(int)

    #Step 12: Find only Home Locations and Return
    only_home_df = actual_locations.where(actual_locations["f2_prob"] > 0.50)
    homes_array = only_home_df["original_location"].as_matrix()
    return homes_array

# # Run Everything Below this Cell before running above cell

# # Trained Models

#Features for models
f1_ground_truth_features = np.load("/Users/frances/Documents/Tribe/ops/InstaLocation_project/ground_truth_features.npy")
f1_ground_truth_response = ground_truth["is_location?"]
f2_ground_truth_features = np.load("/Users/frances/Documents/Tribe/ops/InstaLocation_project/f2_lr_features.npy")
f2_ground_truth_response = f2_lr_filtered["is_home?"]

# Filter 1 Model
f1_lr = LogisticRegression()
f1_lr.fit(f1_ground_truth_features, f1_ground_truth_response)

#Filter 2 Model
f2_lr = LogisticRegression()
f2_lr.fit(f2_ground_truth_features, f2_ground_truth_response)

# Find top 50 features
f1_rf = RandomForestClassifier()
f1_rf.fit(f1_ground_truth_features, f1_ground_truth_response)
f1_importance_arr = f1_rf.feature_importances_
f1_inx = (-f1_importance_arr).argsort()[:50]
f2_rf = RandomForestClassifier()
f2_rf.fit(f2_ground_truth_features, f2_ground_truth_response)
f2_importance_arr = f2_rf.feature_importances_
f2_inx = (-f2_importance_arr).argsort()[:50]


# ## Step 2 (Extract Locations) Little Functions

def extract_location(df, map_column):
    """Adds a column to original df where each cell contains a list of locations found
    in the text of that row
    df: Panda dataframe containing the instagram profile description, as well as ambassador ids.
    map_column: The column name (as a string) that contains the profile text."""

    def whole_compiled_search(compiled_reg, text):
        """Searches for whole-word search terms and outputs any that is found in the text.
        compiled_reg: A compiled regex search object containing either airport codes or
        US state abbreviations.
        text: the text of ONE Instagram profile description."""
        text = re.sub(r'[^\w\s]','',text)
        if isinstance(text, float): #catches NaNs
            return []
        return compiled_reg.findall(text) #return words that match

    def partial_everything_search(text):
        """Searches for partial-word seach terms in the text (such as cities, countries,
        nationalities, or flag emojis), and returns them.
        text: the text of ONE Instagram profile description"""
        def partial_helper(searchterm):
            nonlocal partial_matches
            if searchterm.lower() in text.lower():
                partial_matches.append(searchterm)
        partial_matches = []
        if isinstance(text, float): #catches NaNs
            return []
        else:
            all_partial_locs["Name"].map(partial_helper)# map search function to a list of every location
            return partial_matches

    partial_terms_ser = df[map_column].map(partial_everything_search)\
                                      .rename("partial") #find partial search terms
    airport_ser = df[map_column].map(lambda text:whole_compiled_search(airport_reg, text))\
                                .rename("airport") #find airports
    state_abvs_ser = df[map_column].map(lambda text:whole_compiled_search(upper_state_abv_reg, text))                                   .rename("state_abvs") #find state abbv
    country_abvs_ser = df[map_column].map(lambda text:whole_compiled_search(country_abv_reg, text))                                   .rename("country_abvs") #find country abbv

    placeholder = partial_terms_ser.add(airport_ser)
    second_placeholder = placeholder.add(country_abvs_ser)
    all_together_ser = second_placeholder.add(state_abvs_ser) #combine all locations together in one series
    locations = all_together_ser.rename("matched_location").to_frame() #make into df

    def replace_with_none(locations_list): # consistency
        if locations_list == []:
            locations_list = ["None"]
        return locations_list

    locations["matched_location"] = locations["matched_location"].map(replace_with_none)

    #join "location" column to original df and return
    return df.join(locations).fillna(value = "None") #fillna so next function will work


# ## Step 3 (Expand) Little Functions

#link for solution: https://stackoverflow.com/questions/26068021/iterate-over-rows-and-expand-pandas-dataframe
def expand_location_df(df):
    '''Dataframe must have NaNs filled with some value, preferably "None".
    df: the dataframe returned by the df_with_location function.
    Must have a "location" column that contains a list of locations, or list of "None".'''

    #expands the locations and profile description text
    def expand_descriptions(row):
        locations = row['matched_location'] if isinstance(row['matched_location'], list) else [row['matched_location']]
        s = pd.Series(row['combined_descriptions'], index=list(set(locations)))
        return s

    #formatting
    df_expand_text = df.apply(expand_descriptions, axis=1).stack()
    df_expand_text = df_expand_text.to_frame().reset_index(level=1, drop=False)
    df_expand_text.columns = ['matched_location', 'combined_descriptions']
    df_expand_text.reset_index(drop=True, inplace=True)

    return df_expand_text


# **Code to add to bottom of expand df function if you want to track ambassador name as well**
# #expands the ambassador names
#
#     def expand_name(row):
#         locations = row['matched_location'] if isinstance(row['matched_location'], list) else [row['matched_location']]
#         s = pd.Series(row['ambassador_name'], index=list(set(locations)))
#         return s
#
#     #formatting
#     df_expand_name = df.apply(expand_name, axis=1).stack()
#     df_expand_name = df_expand_name.to_frame().reset_index(level=1, drop=False)
#     df_expand_name.columns = ['matched_location', 'ambassador_name']
#     df_expand_name.reset_index(drop=True, inplace=True)
#     df_expand_name = df_expand_name.drop('matched_location', axis = 1) #drop extra location column to join
#     df_expand_name = df_expand_name["ambassador_name"]
#
#     #join ids with locations and descriptions
#     with_descrip_and_ids = df_expand_text.join(df_expand_ids)
#
#     #join ids with locations and descriptions
#     with_name = with_descrip_and_ids.join(df_expand_name)
#     return with_name


# ## Step 4 (Find original) Little Functions

def find_original_location(location, text):
    """Given a location string, returns how the location is formatted in the text.
    For example, if text is 'I live in PARIsc', and location is 'paris', returns 'PARIsc'.
    Note: the location passed in must have been found in the text passed in!
    text: The text (as a string) of a profile description
    location: a location that Step 2 function found in the above text."""

    if location == "None": #there's no original location
        return location

    elif len(location) > 3: #partial word search
        word_object = re.search(r'\w*(?:{search_term})\w*'.format(search_term=re.escape(location)), text, flags = re.IGNORECASE)
        return word_object[0]

    else: #whole word search-for things like airport codes and state abbvs
        return location #because the location function only finds it if it exactly matches our list


def make_origin_loc_column(df, text_column, location_column):
    """Must use exapanded row df with location column. Returns original df with an
    'Original Location' column, that contains how the 'matched' location is formatted in the text.
    df: dataframe, with expanded rows, that contains the location column from Step 2.
    text_column: The name (as as string) of the column containing the profile descriptions
    location_column: The name (as a string) of the column containing the location found in text."""
    original_loc = []
    for index, row in df.iterrows(): #iterate through rows of df
        text = row[text_column]
        loc = row[location_column] #find paramters
        original_loc.append(find_original_location(loc, text)) #append original loc to list
    original_df = pd.Series(original_loc)\
                    .rename("original_location")\
                    .to_frame() #make list to series to dataframe to add as a column
    return df.join(original_df)


# ## Step 5 (extract context) Little Functions

def extract_context(keyword, text):
    """Returns a string containing the keyword (a location), and the 5 words before and
    after the keyword in the text. If keyword is "None" (ie no location was found), it returns "None"
    Keyword: a location string or 'None'.
    text: One Instagram profile description text, as a string"""
    if keyword is not "None":
        before_keyword, keyword_list, after_keyword = text.partition(keyword)
        #below turns strings to lists of words so we can count indices
        before_keyword, keyword_list, after_keyword = before_keyword.split(), keyword_list.split(), after_keyword.split()
        if len(before_keyword) >= 5:
            left_words_list = before_keyword[len(before_keyword)-5:] #take only 5 words
        else:
            left_words_list = before_keyword #since less than 5 words, take all
        if len(after_keyword) >= 5:
            right_words_list = after_keyword[:5] #take only 5 words
        else:
            right_words_list = after_keyword #since less than 5 words, takes all
        left = " ".join(left_words_list) #turn lists back to strings
        right = " ".join(right_words_list)
        return left + " " + keyword + " " + right #stick everything together
    return keyword #if the location was None, return None (there is no context if a location not found)

def make_context_column(df, text_column, orig_location_column):
    """Must use with the expanded location df (ie such that there is one location per row).
    Returns the original df, with a new 'Context' column, that contains the context of each
    location found (one context per row)
    df: expanded dataframe containing a text column, and the location column from Step 2.
    text_column: The name (as a string) of the column containing the profile descriptions.
    location_column: The name (as a string) of the column containing location found in text"""
    context_list = []
    for index, row in df.iterrows(): #note: internet says itertuples is faster but gives us errors
        text = row[text_column]
        loc = row[orig_location_column]
        context = extract_context(loc, text) #pass parameters to other function
        context_list.append(context) #append context to list
    context_df = pd.Series(context_list)\
                   .rename("Context")\
                   .to_frame() #convert to dataframe for next step
    return df.join(context_df) #add contex column to df.


# # Step 6 (before and After context) Little Functions

def before_context(context, original_location):
    """Context: String from 'Context' column of the dataframe. Is the 5 words before and after
    location found
    original_location: The string from the 'original_location' column- the location found.
    Returns the 5 words before the location"""
    before, loc, after = context.partition(original_location)
    return before

def after_context(context, original_location):
    """Context: String from 'Context' column of the dataframe. Is the 5 words before and after
    location found
    original_location: The string from the 'original_location' column- the location found.
    Returns the 5 words after the location"""
    before, loc, after = context.partition(original_location)
    return after

def make_before_and_after_columns(df, context_column, orig_location_column):
    """Must use with the expanded location df (ie such that there is one location per row).
    Returns the original df, with two columns added- a before context column, and an after
    context column
    df: expanded dataframe containing a text column, and the location column from Step 2.
    text_column: The name (as a string) of the column containing the profile descriptions.
    location_column: The name (as a string) of the column containing location found in text"""
    before_list = []
    after_list = []
    for index, row in df.iterrows(): #note: internet says itertuples is faster but gives us errors
        contxt = row[context_column]
        loc = row[orig_location_column]
        before = before_context(contxt, loc) #pass parameters to other function
        before_list.append(before) #append before_context to list
        after = after_context(contxt, loc)
        after_list.append(after)
    before_context_df = pd.Series(before_list)\
                          .rename("before_loc")\
                          .to_frame()
    after_context_df = pd.Series(after_list)\
                         .rename("after_loc")\
                         .to_frame()
    split_context_df = before_context_df.join(after_context_df)
    return df.join(split_context_df) #add contex column to df.


# # Step 7 (Filter 1 Features) Little Functions:

def return_feature_array_for_row(row):
    """Returns a array of features for that one row.
    These features only need the row to be calculated.
    Different for each location candidate
    Will be applied to the df returned by ground truth functions"""
    #setup paramters/variables
    keyword = row["matched_location"]
    text = row['combined_descriptions']
    original = row['original_location']
    context = row['Context']
    common_abvs = ["CA", "UK", "US", "NY", "LA"]
    uncommon_abvs = ["PR", "IN", "ME", "SO", "DM", "IT", "SC", "BY", "AT", "OR", "TV", "ET", "TO"]
    pushpin = int("📍" in context)
    live = fast_search("live", context)
    based = fast_search("based", context)
    is_from = fast_search("from", context) #filter 2
    airplane = int("✈️" in context) #filter 2
    is_next = fast_search("next", context)# filter 2
    born = fast_search("born", context)#filter 2
    currently = fast_search("currently", context)
    #features
    len_key_matches_orig = compare_len_of_locs(keyword, original)
    punc_matches = int(keyword == original)
    len_keyword = len(keyword)
    num_spaces_keyword = len(keyword.split()) - 1
    good_indicators = surrounding_good(pushpin, live, based, currently)
    bad_indicators = surrounding_bad(is_from, airplane, is_next, born)
    emoji_or_not = is_emoji(keyword)
    is_common = fast_search(keyword, common_abvs)
    is_uncommon = fast_search(keyword, uncommon_abvs)
    is_whole = is_whole_word(text, keyword) #trying to catch word-within-a-word problems
    num_upper_words = count_upper_words(context, original)
    model = int("model" in context)
    return [len_key_matches_orig, punc_matches, len_keyword, num_spaces_keyword,\
            good_indicators, bad_indicators, emoji_or_not,is_common, is_uncommon,\
            is_whole, num_upper_words, model]


# ** Feature to add to filter one features funtion above if you want to track ambassador name**
# same_as_name = int(original in row["ambassador_name"])
# **Add above code after "model"

def df_features(df):
    """Calculates two features that require the whole dataframe to calculate.
    Will be the same for every location found in a single piece of text.
    Df: The dataframe returned after step 6"""
    num_other_locations = len(df)
    num_flag_emojis = fast_emoji_search(df)
    return [num_other_locations, num_flag_emojis]

def return_features_list(df):
    """Applies the return_feature_array_for_row and returns a array of lists, so one list of
    features for each row"""
    return df.apply(return_feature_array_for_row, axis = 1)

def return_df_features(df_features_lst, df):
    """Takes the list of 2 features returned by the df_features function, and dupilcates the
    values for each location found in the text (since these features are same for each location)
    Puts them into proper format to be combined with the other 2 sets of features calculated"""
    new_long_lst = []
    for i in range(len(df)):
        new_long_lst.append(df_features_lst)
    return np.array(new_long_lst)

def combine_feature_lists(our_features_ser, cv_sparse_matrix, df_features_arr):
    """Combines 3 sets of feature arrays into one array, so that each location candidate has
    all its features in one place.
    our_features_ser: The features returned by return_feature_array_for_row, ie the features
    we created that only need to access one row
    cv_sparse_matrix: The sparse matrix of features returned by a count vectorizer (more features)
    df_features_arr: The features returned by return_df_features (ie the features we created that
    need to access the whole df)
    Returns: a numpy array that will be passed to models."""

    #reformat our features list into a df with each feature as its own column
    #link for source: https://chrisalbon.com/python/pandas_expand_cells_containing_lists.html
    our_features_df = our_features_ser.rename('features').to_frame()
    our_features = our_features_df["features"].apply(pd.Series).as_matrix()

    #reformat sparse matrix:
    cv_sparse_to_array = np.array(cv_sparse_matrix.todense())

    #return array of all features
    return np.hstack((our_features, cv_sparse_to_array, df_features_arr))

def fast_search(keyword, sequence):
    """A faster way to see if our keywords are in the context.
    keyword: A string of indicator word, like 'live' or 'from'.
    Keyword CANNOT be an emoji-those don't work with sets. Use 'in' method seperately
    sequence: Either a string (like if searching the context), or a list (like if looking
    at abbreviations)"""
    if isinstance(sequence, str):
        keyword, sequence = re.sub(r'[^\w\s]','',keyword), re.sub(r'[^\w\s]','', sequence)
        keyword, sequence = keyword.lower(), sequence.lower()
        keyword_set = set([keyword])
        sequence_set = set(sequence.split())
        return np.count_nonzero(keyword_set.intersection(sequence_set))
    else: #list - no splitting needed
        keyword_set = set([keyword])
        sequence_set = set(sequence)
        return np.count_nonzero(keyword_set.intersection(sequence_set))

def f1_make_cv_word_features(context):
    """Creates a Count Vectorizer to create features
    Context: The whole (ie 10 word) context from the dataframe's 'Context' column
    Returns:  a sparse matrix of features"""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(
        ngram_range = (1, 3)
    )
    return cv.fit_transform(context)

def fast_emoji_search(df):
    """Calculates a feature for feature lists.
    Counts how many of the location canidates are flag emojis
    returns an int of count."""
    count = 0
    for index, row in df.iterrows():
        if re.search('[a-zA-Z]', row['matched_location']) == None:
            count += 1
    return count

def count_upper_words(context, original_loc):
    """Returns the number of all-caps words in the context of a location. Designed to catch
    country 'abbreviations' that weren't actually locations but just someone saying a bunch
    of capitalized stuff"""
    if len(original_loc) < 3:
        word_list = context.split()
        upper_list = [word for word in word_list if word.isupper()]
        return len(upper_list)
    else:
        return 0

def surrounding_good(f1, f2, f3, f4):
    """Returns 1 if the keyword is surrounded by one of the good location indicators
    0 if not. The 'good' indicators are pushpin, live, and based"""
    if f1 or f2 or f3 or f4== 1:
        return 1
    else:
        return 0
def surrounding_bad(f4, f5, f6, f7):
    """Returns 1 if the keyword is surrounded by one of the bad location indicators, 0 otherwise
    Bad indicators are based, airplane, and next, bc these signal non-home locations (for filter 2)"""
    if f4 or f5 or f6 or f7 == 1:
        return 1
    else:
        return 0

#link: https://stackoverflow.com/questions/5319922/python-check-if-word-is-in-a-string
def is_whole_word(text, word):
    """Checks to see if word is a whole word, with spaces before and after.
    Deals with word-within-words problem.
    Returns 0 if word is within another word
    Returns 1 if word is its own seperate whole word"""
    search_object = re.search(r'\b({0})\b'.format(word), text, flags=re.IGNORECASE)
    if search_object == None:
        return 0
    else:
        return 1

def compare_len_of_locs(matched, original):
    """Returns whether the length of the matched location is the same as length of
    the original. Designed to catch things that aren't actually locations. Like 'erie'
    coming from 'experiences'"""
    matched, original = re.sub(r'[^\w\s]','',matched), re.sub(r'[^\w\s]','',original) #remove punc
    matched, original = matched.lower(), original.lower()
    if matched == original:
        return 1
    else:
        return 0

def is_emoji(matched_location):
    """whether or not the location is an emoji
    Returns 1 if emoji, 0 if not"""
    if re.search('[a-zA-Z]', matched_location) == None:
        return 1
    return 0


# # Step 10 (Filter 2 features) Little Functions:

def f2_return_feature_array_for_row(row):
    """Returns a array of features for that one row.
    Will be applied to the df returned by ground truth functions"""
    #setup paramters/variables
    keyword = row["matched_location"]
    text = row['combined_descriptions']
    original = row['original_location']
    context = row['Context']
    before_loc = row["before_loc"]
    after_loc = row['after_loc']
    #features
    pushpin_b4 = int("📍" in before_loc)
    live_b4 = fast_search("live", before_loc)
    living_b4 = fast_search("living", before_loc)
    living_after = fast_search("living", after_loc)
    based_b4 = fast_search("based", before_loc)
    based_after = fast_search("based", after_loc)
    is_from_b4 = fast_search("from", before_loc) #filter 2
    airplane_b4 = int("✈️" in before_loc) #filter 2
    airplane_after = int("✈️" in after_loc) #filter 2
    is_next_b4 = fast_search("next", before_loc)# filter 2
    born_b4 = fast_search("born", before_loc)#filter 2
    via_b4 = fast_search("via", before_loc) #filter 2
    soon_emoji_b4 = int("🔜" in before_loc)#filter 2
    currently_b4 = fast_search("currently", before_loc)
    located_b4 = fast_search("located", before_loc)
    emoji_or_not = is_emoji(keyword)
    nationality_or_not = is_ethnicity(keyword)
    f1_result = row["f1_prob"]
    living_in_after = fast_search("living in", after_loc)
    based_in_after = fast_search("based in", after_loc)
    in_after = fast_search("in", after_loc)
    in_before = fast_search("in", before_loc)
    college_context = fast_search("college", context)
    soon_context = fast_search("soon", context)
    model = int("model" in context)

    return [pushpin_b4, live_b4, living_b4, living_after, based_b4, based_after, is_from_b4,\
            airplane_b4, airplane_after, is_next_b4, born_b4, via_b4, soon_emoji_b4,\
            currently_b4, located_b4, emoji_or_not, nationality_or_not,\
            f1_result, living_in_after, based_in_after, in_after, in_before,\
            college_context, soon_context, model]

def f2_return_features_list(df):
    """Applies the return_feature_array_for_row to each row and returns a array of lists,
    so one list of features for each row"""
    ser = df.apply(f2_return_feature_array_for_row, axis = 1)
    return ser

def is_ethnicity(matched_location):
    """Checks to see if the location in the 'matched_location' column is an ethnicity.
    Hopefully helps filter 2 distinguish home locations from origins.
    Returns 1 if matched_location is an ethnicity, 0 if not."""
    return int(matched_location in set_ethnicities)

def f2_make_cv_word_features(context):
    """Creates a Count Vectorizer to create features
    Context: The whole (ie 10 word) context from the dataframe's 'Context' column
    Returns:  a sparse matrix of features"""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(
        ngram_range = (1, 3)
    )
    return cv.fit_transform(context)



# # How We Generated our Training Data

# * First, we made several SAPD files (from RAF). In searching for profiles, we sometimes used neutral search terms (such as "fragrance" or ":earth_americas:") -- these search terms did not imply a specific location and thus provided us with a variety of geographic locations. Other times, we used more specific search terms like "Mali" or "Los Angeles" in order to further analyze the errors that come up with specific locations.
# * Second, we put all the SAPD scrapes into one file (about 825 unique influencers).
# * Third, we ran steps 2 through 6 in the "everything" function, to create a set up for ground truth
#      * The file needed to have one location per row, have matched_location, original_location, combined_descriptions, Context, before_loc, after_loc, and ambassador id columns
# * Fourth, we exported that set up file to Google Sheets, where we deleted the index column (a byproduct of exporting a panda dataframe), and added an "is_location?"(filter 1) column and an "is_home?" (filter 2) column.
# * Fifth, we manually went through all 2011 rows and read the profile description, and manually marked whether the candidate was a location (1) or not location (0) and whether it was a home location (1) or not a home location (0).
#     * This is not necessary for most use of the function, because you want a machine to do it for you. This step is only necessary if you want to create more ground truth training data for the models
# * Sixth, we took the ground_truth file (with our manual checking), and completed steps 7 through 9 in the "everything" model above. (ie filter 1 features and model)
#      * Since we were testing 3 different models, we created 3 different "filtered locations" files (ie we created an svc_filtered, an rf_filtered, and an lr_filtered file, each containing only the candidates where the model was more than 50% confident that it was a location. Since each model had different results, we wanted to preserve that in creating filtered df). However, in the "everything" function, we only trained a logistic regression model, since that seemed to be the most accurate in our testing.
# * Seventh, completed steps 10 through 11 (ie filter 2 features and model) on the filtered datasets(ie only on actual locations).

# # Future Improvements Notes

# * For Proper Names problem:
#     * Though in our groundtruths we did not keep ambassador names, it will be beneficial to use this information in determining whether a person is mentioning a proper name or a location. Also, adding a list of family words and checking if any of those words appear in the context would help solve the problem of people writing phrases like "Mommy to Austin." Some family words that we have seen are:
#         * mom; mommy; grandma; children; son; daughter; sons; daughters
#     * Additionally, since some names are common in posts about lifestyle brands, it could help to hardcode out some specific names (like Elizabeth and James) to not be confused with the corresponding locations (Elizabeth in Indiana)
# * More Location Indicator words:
#     * house emoji "🏠"
#     * the word "home"
# * Demonyms:
#     * Currently, we are using a csv of nationalities instead of demonyms. Expanding this to include all types of demonyms could help catch more locations. We have included a csv of demonyms in the github repository in case you would like to use this in the future.
#     * However, it is also important to note the problem of people mentioning a demonym when they actually live somewhere else (for example, we sometimes saw influencers write "Irish in England")
#         * Currently, to approach this problem, we have a feature in Filter 2 that notes whether or not a location is an ethnicity.
#         * Other ideas include:
#             * Doing more detailed analyses on location indicator words like "in" and "living in" -- oftentimes people would write phrases like "French in Italy" or "Irish living in Spain." Even after adding our "before context" and "after context" analyses, some of these errors are still included in the final results. To improve this, one idea is to see what word precedes or follows the location indicator word. In the above examples, we have an ethnicity followed by a non-demonym country mention. Since the location indicator words imply that the second location is the home, we can be fairly confident that the first mention is not a home while the second one is.
#             * Thus, for demonyms, we can do more comparative analyses -- if other locations are mentioned in the text, we would want to see where these locations are mentioned and if they are preceded or followed by location indicator words. If the model identifies those other locations as more likely to be the home, then the probability of the demonym being a home should decrease.
# * More comparative approaches:
#     * This follows to a more general improvement -- rather than only looking at locations as individual location mentions, further steps should attempt to also look at all location mentions collectively for each influencer. This will be especially helpful for models and world travelers. Below is one example of a world traveler's profile description:
# "📍: 🇳🇿
# ✈️: London, Portugal, Spain and Luxembourg soon
# ❤️:☀️🍾☕️ website: http://www.sarahseestheworld.com/;Passionate about travelling and returning home to Wellington. I also heart wifi and good coffee. Instagram @sarahseestworld"
#         * To analyze this above description, we can look at all the location words comparatively ("🇳🇿; London; Portugal; Spain; Luxembourg; Wellington"), rather than independently. One way to do this is to tag each location based on three criteria:
#             * the location's type (demonym, emoji, country, city, etc.)
#             * the location's corresponding indicator words (pushpin emoji, airplane emoji, the word "soon", the word "home", etc.)
#             * the synonymous locations: in other words, the flag emoji (if applicable) and the broader location names that a person is also located in if they are located in a certain city, region, or state -- Wellington, for example, would be tagged with "🇳🇿" and "New Zealand"
#         * For the above example, the tags would be as follows:
#             *  🇳🇿 : type = emoji; indicators = 📍; synonyms = New Zealand
#             *  London: type = city; indicators = ✈️, "soon"; synonyms = UK, 🇬🇧
#             *  [similar tags for Portugal, Spain, Luxembourg]
#             *  Wellington: type = city; indicators = "home"; synonyms = New Zealand, 🇳🇿
#         * For Filter 2 specifically, a hierarchy could be established in which certain location indicators have higher probabilites than other words in indicating home locations. Two examples that show up here are that the pushpin emoji and the word "home" are often more likely to indicate homes than are the airplane emoji and the word "soon." By establishing this hierarchy, home identification can become more accurate. Other weighting schemes can also be established for the other tags, like the location type category. Demonyms tend to be the least likely to be home locations, so these can be placed as lower likelihood under the other data types. Finally, the synonyms can be useful in matching different ways of writing the same location together. If a person uses 🇳🇿 and "Wellington" in one description, for example, these locations match (same synonyms) and could receive higher probability.
# * Context:
#     * Taking a more quantitative approach with regards to how many words to analyze before and after the location word can help with improving accuracy. Also, identifying whether or not other locations are mentioned in the context would be good. In this way, the problem of a location indicator appearing near a location word but not actually referencing that location word can be alleviated.
