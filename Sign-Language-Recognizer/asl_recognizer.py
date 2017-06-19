
# coding: utf-8

# # Artificial Intelligence Engineer Nanodegree - Probabilistic Models
# ## Project: Sign Language Recognition System
# - [Introduction](#intro)
# - [Part 1 Feature Selection](#part1_tutorial)
#     - [Tutorial](#part1_tutorial)
#     - [Features Submission](#part1_submission)
#     - [Features Unittest](#part1_test)
# - [Part 2 Train the models](#part2_tutorial)
#     - [Tutorial](#part2_tutorial)
#     - [Model Selection Score Submission](#part2_submission)
#     - [Model Score Unittest](#part2_test)
# - [Part 3 Build a Recognizer](#part3_tutorial)
#     - [Tutorial](#part3_tutorial)
#     - [Recognizer Submission](#part3_submission)
#     - [Recognizer Unittest](#part3_test)
# - [Part 4 (OPTIONAL) Improve the WER with Language Models](#part4_info)

# <a id='intro'></a>
# ## Introduction
# The overall goal of this project is to build a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models.  In particular, this project employs  [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).  In this video, the right-hand x and y locations are plotted as the speaker signs the sentence.
# [![ASLR demo](http://www-i6.informatik.rwth-aachen.de/~dreuw/images/demosample.png)](https://drive.google.com/open?id=0B_5qGuFe-wbhUXRuVnNZVnMtam8)
# 
# The raw data, train, and test sets are pre-defined.  You will derive a variety of feature sets (explored in Part 1), as well as implement three different model selection criterion to determine the optimal number of hidden states for each word model (explored in Part 2). Finally, in Part 3 you will implement the recognizer and compare the effects the different combinations of feature sets and model selection criteria.  
# 
# At the end of each Part, complete the submission cells with implementations, answer all questions, and pass the unit tests.  Then submit the completed notebook for review!

# <a id='part1_tutorial'></a>
# ## PART 1: Data
# 
# ### Features Tutorial
# ##### Load the initial database
# A data handler designed for this database is provided in the student codebase as the `AslDb` class in the `asl_data` module.  This handler creates the initial [pandas](http://pandas.pydata.org/pandas-docs/stable/) dataframe from the corpus of data included in the `data` directory as well as dictionaries suitable for extracting data in a format friendly to the [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) library.  We'll use those to create models in Part 2.
# 
# To start, let's set up the initial database and select an example set of features for the training set.  At the end of Part 1, you will create additional feature sets for experimentation. 

# In[1]:

import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame


# In[2]:

asl.df.ix[98,1]  # look at the data available for an individual frame


# The frame represented by video 98, frame 1 is shown here:
# ![Video 98](http://www-i6.informatik.rwth-aachen.de/~dreuw/database/rwth-boston-104/overview/images/orig/098-start.jpg)

# ##### Feature selection for training the model
# The objective of feature selection when training a model is to choose the most relevant variables while keeping the model as simple as possible, thus reducing training time.  We can use the raw features already provided or derive our own and add columns to the pandas dataframe `asl.df` for selection. As an example, in the next cell a feature named `'grnd-ry'` is added. This feature is the difference between the right-hand y value and the nose y value, which serves as the "ground" right y value. 

# In[3]:

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary


# ##### Try it!

# In[4]:

from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations

asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# test the code
test_features_tryit(asl)


# In[5]:

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]


# ##### Build the training set
# Now that we have a feature list defined, we can pass that list to the `build_training` method to collect the features for all the words in the training set.  Each word in the training set has multiple examples from various videos.  Below we can see the unique words that have been loaded into the training set:

# In[6]:

training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))


# The training data in `training` is an object of class `WordsData` defined in the `asl_data` module.  in addition to the `words` list, data can be accessed with the `get_all_sequences`, `get_all_Xlengths`, `get_word_sequences`, and `get_word_Xlengths` methods. We need the `get_word_Xlengths` method to train multiple sequences with the `hmmlearn` library.  In the following example, notice that there are two lists; the first is a concatenation of all the sequences(the X portion) and the second is a list of the sequence lengths(the Lengths portion).

# In[7]:

training.get_word_Xlengths('CHOCOLATE')


# ###### More feature sets
# So far we have a simple feature set that is enough to get started modeling.  However, we might get better results if we manipulate the raw values a bit more, so we will go ahead and set up some other options now for experimentation later.  For example, we could normalize each speaker's range of motion with grouped statistics using [Pandas stats](http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats) functions and [pandas groupby](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html).  Below is an example for finding the means of all speaker subgroups.

# In[8]:

df_means = asl.df.groupby('speaker').mean()
df_means


# To select a mean that matches by speaker, use the pandas [map](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) method:

# In[9]:

asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
asl.df.head()


# ##### Try it!

# In[10]:

from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker

df_std = asl.df.groupby('speaker').std()

# test the code
test_std_tryit(df_std)


# <a id='part1_submission'></a>
# ### Features Implementation Submission
# Implement four feature sets and answer the question that follows.
# - normalized Cartesian coordinates
#     - use *mean* and *standard deviation* statistics and the [standard score](https://en.wikipedia.org/wiki/Standard_score) equation to account for speakers with different heights and arm length
#     
# - polar coordinates
#     - calculate polar coordinates with [Cartesian to polar equations](https://en.wikipedia.org/wiki/Polar_coordinate_system#Converting_between_polar_and_Cartesian_coordinates)
#     - use the [np.arctan2](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.arctan2.html) function and *swap the x and y axes* to move the $0$ to $2\pi$ discontinuity to 12 o'clock instead of 3 o'clock;  in other words, the normal break in radians value from $0$ to $2\pi$ occurs directly to the left of the speaker's nose, which may be in the signing area and interfere with results.  By swapping the x and y axes, that discontinuity move to directly above the speaker's head, an area not generally used in signing.
# 
# - delta difference
#     - as described in Thad's lecture, use the difference in values between one frame and the next frames as features
#     - pandas [diff method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html) and [fillna method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html) will be helpful for this one
# 
# - custom features
#     - These are your own design; combine techniques used above or come up with something else entirely. We look forward to seeing what you come up with! 
#     Some ideas to get you started:
#         - normalize using a [feature scaling equation](https://en.wikipedia.org/wiki/Feature_scaling)
#         - normalize the polar coordinates
#         - adding additional deltas
# 

# In[13]:

# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
cols = ['right-x', 'right-y','left-x', 'left-y']

for i,f in enumerate(features_norm):
    mean = asl.df['speaker'].map(df_means[cols[i]])
    std = asl.df['speaker'].map(df_std[cols[i]])
    asl.df[f]=(asl.df[cols[i]] - mean) / std


# In[14]:

# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

asl.df['polar-rr'] = ((asl.df['grnd-rx']**2) + (asl.df['grnd-ry']**2))**(0.5)
asl.df['polar-lr'] = ((asl.df['grnd-lx']**2) + (asl.df['grnd-ly']**2))**(0.5)

asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])


# In[15]:

# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)


# In[16]:

# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like

# TODO define a list named 'features_custom' for building the training set

# The difference between the movements of the left and right hands
# This should help determine if they are performing the same action or not

# Multiply delta-lx to correct for the inverse x-axis movements of the left hand compared to the right hand.
asl.df['delta-rlx'] = asl.df['delta-rx'] - -1*asl.df['delta-lx']
asl.df['delta-rly'] = asl.df['delta-ry'] - asl.df['delta-ly']

# Calculate the Euclidean distance between the hands

asl.df['dist_hands'] = np.sqrt((asl.df['right-x'] - asl.df['left-x'])**2 + (asl.df['right-y'] - asl.df['left-y'])**2)

# Scale the x and y values to between 0 and 1

df_maxs = asl.df.groupby('speaker').max()
df_mins = asl.df.groupby('speaker').min()

max_rx = asl.df['speaker'].map(df_maxs['right-x'])
max_ry = asl.df['speaker'].map(df_maxs['right-y'])
max_lx  = asl.df['speaker'].map(df_maxs['left-x'])
max_ly  = asl.df['speaker'].map(df_maxs['left-y'])

min_rx = asl.df['speaker'].map(df_mins['right-x'])
min_ry = asl.df['speaker'].map(df_mins['right-y'])
min_lx  = asl.df['speaker'].map(df_mins['left-x'])
min_ly  = asl.df['speaker'].map(df_mins['left-y'])

asl.df['scaled-rx'] = (asl.df['right-x'] - min_rx) / (max_rx - min_rx)
asl.df['scaled-ry'] = (asl.df['right-y'] - min_ry) / (max_ry - min_ry)
asl.df['scaled-lx'] = (asl.df['left-x'] - min_lx) / (max_lx - min_lx)
asl.df['scaled-ly'] = (asl.df['left-y'] - min_ly) / (max_ly - min_ly)

# Create list of custom features
features_custom = ['delta-rlx','delta-rly','dist_hands','scaled-rx','scaled-ry','scaled-lx','scaled-ly']


# **Question 1:**  What custom features did you choose for the features_custom set and why?
# 
# **Answer 1:**  
# The delta features were added to detect the similarity between the movements of the left and right hands. This should make it easier for the model to understand when two-handed signs with identical movements are being made, such as 'cat'.
# 
# The dist_hands feature was added to detect the distance between the hands. Some signs involve hands touching each other, at which point this value should drop to zero. This should make it easier for the model to understand when signs involving both hands, with close interaction are being made, such as 'name'.
# 
# The scaled features will scale the x and y values to between 0 and 1, based on their minimum and maximum values. Although this is similar to the norm features, new information can still be learned by the model because it regularizes the x and y values to each speaker's upper or lower limit.

# <a id='part1_test'></a>
# ### Features Unit Testing
# Run the following unit tests as a sanity check on the defined "ground", "norm", "polar", and 'delta"
# feature sets.  The test simply looks for some valid values but is not exhaustive.  However, the project should not be submitted if these tests don't pass.

# In[17]:

import unittest
# import numpy as np

class TestFeatures(unittest.TestCase):

    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))
                         
suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)


# <a id='part2_tutorial'></a>
# ## PART 2: Model Selection
# ### Model Selection Tutorial
# The objective of Model Selection is to tune the number of states for each word HMM prior to testing on unseen data.  In this section you will explore three methods: 
# - Log likelihood using cross-validation folds (CV)
# - Bayesian Information Criterion (BIC)
# - Discriminative Information Criterion (DIC) 

# ##### Train a single word
# Now that we have built a training set with sequence data, we can "train" models for each word.  As a simple starting example, we train a single word using Gaussian hidden Markov models (HMM).   By using the `fit` method during training, the [Baum-Welch Expectation-Maximization](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) (EM) algorithm is invoked iteratively to find the best estimate for the model *for the number of hidden states specified* from a group of sample seequences. For this example, we *assume* the correct number of hidden states is 3, but that is just a guess.  How do we know what the "best" number of states for training is?  We will need to find some model selection technique to choose the best parameter.

# In[18]:

import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))


# The HMM model has been trained and information can be pulled from the model, including means and variances for each feature and hidden state.  The [log likelihood](http://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution) for any individual sample or group of samples can also be calculated with the `score` method.

# In[19]:

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()
    
show_model_stats(demoword, model)


# ##### Try it!
# Experiment by changing the feature set, word, and/or num_hidden_states values in the next cell to see changes in values.  

# In[20]:

my_testword = 'CHOCOLATE'
model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
show_model_stats(my_testword, model)
print("logL = {}".format(logL))


# ##### Visualize the hidden states
# We can plot the means and variances for each state and feature.  Try varying the number of states trained for the HMM model and examine the variances.  Are there some models that are "better" than others?  How can you tell?  We would like to hear what you think in the classroom online.

# In[21]:

get_ipython().magic('matplotlib inline')


# In[22]:

import math
from matplotlib import (cm, pyplot as plt, mlab)

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()
        
visualize(my_testword, model)


# #####  ModelSelector class
# Review the `ModelSelector` class from the codebase found in the `my_model_selectors.py` module.  It is designed to be a strategy pattern for choosing different model selectors.  For the project submission in this section, subclass `SelectorModel` to implement the following model selectors.  In other words, you will write your own classes/functions in the `my_model_selectors.py` module and run them from this notebook:
# 
# - `SelectorCV `:  Log likelihood with CV
# - `SelectorBIC`: BIC 
# - `SelectorDIC`: DIC
# 
# You will train each word in the training set with a range of values for the number of hidden states, and then score these alternatives with the model selector, choosing the "best" according to each strategy. The simple case of training with a constant value for `n_components` can be called using the provided `SelectorConstant` subclass as follow:

# In[23]:

from my_model_selectors import SelectorConstant

training = asl.build_training(features_delta)  # Experiment here with different feature sets defined in part 1
word = 'FRIEND' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))


# ##### Cross-validation folds
# If we simply score the model with the Log Likelihood calculated from the feature sequences it has been trained on, we should expect that more complex models will have higher likelihoods. However, that doesn't tell us which would have a better likelihood score on unseen data.  The model will likely be overfit as complexity is added.  To estimate which topology model is better using only the training data, we can compare scores using cross-validation.  One technique for cross-validation is to break the training set into "folds" and rotate which fold is left out of training.  The "left out" fold scored.  This gives us a proxy method of finding the best model to use on "unseen data". In the following example, a set of word sequences is broken into three folds using the [scikit-learn Kfold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) class object. When you implement `SelectorCV`, you will use this technique.

# In[24]:

from sklearn.model_selection import KFold

training = asl.build_training(features_ground) # Experiment here with different feature sets
word = 'CHOCOLATE' # Experiment here with different words
word_sequences = training.get_word_sequences(word)
split_method = KFold()
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds


# **Tip:** In order to run `hmmlearn` training using the X,lengths tuples on the new folds, subsets must be combined based on the indices given for the folds.  A helper utility has been provided in the `asl_utils` module named `combine_sequences` for this purpose.

# ##### Scoring models with other criterion
# Scoring model topologies with **BIC** balances fit and complexity within the training set for each word.  In the BIC equation, a penalty term penalizes complexity to avoid overfitting, so that it is not necessary to also use cross-validation in the selection process.  There are a number of references on the internet for this criterion.  These [slides](http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf) include a formula you may find helpful for your implementation.
# 
# The advantages of scoring model topologies with **DIC** over BIC are presented by Alain Biem in this [reference](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf) (also found [here](https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf)).  DIC scores the discriminant ability of a training set for one word against competing words.  Instead of a penalty term for complexity, it provides a penalty if model liklihoods for non-matching words are too similar to model likelihoods for the correct word in the word set.

# <a id='part2_submission'></a>
# ### Model Selection Implementation Submission
# Implement `SelectorCV`, `SelectorBIC`, and `SelectorDIC` classes in the `my_model_selectors.py` module.  Run the selectors on the following five words. Then answer the questions about your results.
# 
# **Tip:** The `hmmlearn` library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.

# In[25]:

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit


# In[75]:

# TODO: Implement SelectorCV in my_model_selector.py
from my_model_selectors import SelectorCV

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))


# In[56]:

# TODO: Implement SelectorBIC in module my_model_selectors.py
from my_model_selectors import SelectorBIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))


# In[57]:

# TODO: Implement SelectorDIC in module my_model_selectors.py
from my_model_selectors import SelectorDIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))


# **Question 2:**  Compare and contrast the possible advantages and disadvantages of the various model selectors implemented.
# 
# **Answer 2:**
# 
# ### CV Selector
# - Advantages:
#  - It's more data efficient since no data needs to be withheld during training to track its performance.
# - Disadvantages: 
#  - Depending on the number of folds, this can really increasing the training time.
# 
# ### BIC Selector
# - Advantages:
#  - Limits overfitting by penalizing complexity.
#  - As seen by the times above, it's the fastest to train.
# - Disadvantages: 
#  - Typically not as accurate as CV.
#  - Will often require more data than CV because it needs to withhold more data for validating the quality of the training. 
# 
# ### DIC Selector
# - Advantages:
#  - [Outperforms BIC](https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf).
#  - Easy to calculate from samples generated by a Markov chain Monte Carlo simulation.
# - Disadvantages: 
#  - Can be more computationally expensive than BIC as it might use more states.

# <a id='part2_test'></a>
# ### Model Selector Unit Testing
# Run the following unit tests as a sanity check on the implemented model selectors.  The test simply looks for valid interfaces  but is not exhaustive. However, the project should not be submitted if these tests don't pass.

# In[58]:

from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)


# <a id='part3_tutorial'></a>
# ## PART 3: Recognizer
# The objective of this section is to "put it all together".  Using the four feature sets created and the three model selectors, you will experiment with the models and present your results.  Instead of training only five specific words as in the previous section, train the entire set with a feature set and model selector strategy.  
# ### Recognizer Tutorial
# ##### Train the full training set
# The following example trains the entire set with the example `features_ground` and `SelectorConstant` features and model selector.  Use this pattern for you experimentation and final submission cells.
# 
# 

# In[59]:

# autoreload for automatically reloading changes made in my_model_selectors and my_recognizer
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from my_model_selectors import SelectorConstant

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorConstant)
print("Number of word models returned = {}".format(len(models)))


# ##### Load the test set
# The `build_test` method in `ASLdb` is similar to the `build_training` method already presented, but there are a few differences:
# - the object is type `SinglesData` 
# - the internal dictionary keys are the index of the test word rather than the word itself
# - the getter methods are `get_all_sequences`, `get_all_Xlengths`, `get_item_sequences` and `get_item_Xlengths`

# In[60]:

test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))


# <a id='part3_submission'></a>
# ### Recognizer Implementation Submission
# For the final project submission, students must implement a recognizer following guidance in the `my_recognizer.py` module.  Experiment with the four feature sets and the three model selection methods (that's 12 possible combinations). You can add and remove cells for experimentation or run the recognizers locally in some other way during your experiments, but retain the results for your discussion.  For submission, you will provide code cells of **only three** interesting combinations for your discussion (see questions below). At least one of these should produce a word error rate of less than 60%, i.e. WER < 0.60 . 
# 
# **Tip:** The hmmlearn library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.

# In[61]:

# TODO implement the recognize method in my_recognizer
from my_recognizer import recognize
from asl_utils import show_errors


# In[66]:

# TODO Choose a feature set and model selector
features = features_custom 
model_selector = SelectorCV 

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)


# In[70]:

# TODO Choose a feature set and model selector
features = features_custom 
model_selector = SelectorBIC 

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)


# In[72]:

# TODO Choose a feature set and model selector
features = features_polar 
model_selector = SelectorDIC 

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)


# **Question 3:**  Summarize the error results from three combinations of features and model selectors.  What was the "best" combination and why?  What additional information might we use to improve our WER?  For more insight on improving WER, take a look at the introduction to Part 4.
# 
# **Answer 3:**
# 
# Despite what was written in [this paper](https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf), BIC Selector achieved the best results in terms of lowest overall WER and average WER. The best results used my customer features. CV Selector performed the worse out of the three selectors and the best set of features was my custom set, which scored an average WER of 0.5243. I expect that BIC achieved the best results because of its ability to regularize and avoid overfitting.
# 
# To improve our WER, it could be useful to compute the probability of word+1 following word using bi-grams or a statistical language model. 
# 
# #### CV Selector
# - features_norm: 0.6685
# - features_polar: 0.6067
# - features_delta: 0.7135
# - features_custom: 0.5562
# - average: 0.6362
# 
# #### BIC Selector
# - features_norm: 0.6124
# - features_polar: 0.5449
# - features_delta: 0.6236
# - **features_custom: 0.4775**
# - average: 0.5646
# 
# #### DIC Selector
# - features_norm: 0.5955
# - features_polar: 0.5281
# - features_delta: 0.6180
# - features_custom: 0.5393
# - average: 0.5702
# 
# #### Average WER for each set of features
# - features_norm: 0.6255
# - features_polar: 0.5599
# - features_delta: 0.6255
# - **feautres_custom: 0.5243**

# <a id='part3_test'></a>
# ### Recognizer Unit Tests
# Run the following unit tests as a sanity check on the defined recognizer.  The test simply looks for some valid values but is not exhaustive. However, the project should not be submitted if these tests don't pass.

# In[36]:

from asl_test_recognizer import TestRecognize
suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
unittest.TextTestRunner().run(suite)


# <a id='part4_info'></a>
# ## PART 4: (OPTIONAL)  Improve the WER with Language Models
# We've squeezed just about as much as we can out of the model and still only get about 50% of the words right! Surely we can do better than that.  Probability to the rescue again in the form of [statistical language models (SLM)](https://en.wikipedia.org/wiki/Language_model).  The basic idea is that each word has some probability of occurrence within the set, and some probability that it is adjacent to specific other words. We can use that additional information to make better choices.
# 
# ##### Additional reading and resources
# - [Introduction to N-grams (Stanford Jurafsky slides)](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)
# - [Speech Recognition Techniques for a Sign Language Recognition System, Philippe Dreuw et al](https://www-i6.informatik.rwth-aachen.de/publications/download/154/Dreuw--2007.pdf) see the improved results of applying LM on *this* data!
# - [SLM data for *this* ASL dataset](ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-boston-104/lm/)
# 
# ##### Optional challenge
# The recognizer you implemented in Part 3 is equivalent to a "0-gram" SLM.  Improve the WER with the SLM data provided with the data set in the link above using "1-gram", "2-gram", and/or "3-gram" statistics. The `probabilities` data you've already calculated will be useful and can be turned into a pandas DataFrame if desired (see next cell).  
# Good luck!  Share your results with the class!

# In[37]:

# create a DataFrame of log likelihoods for the test word items
df_probs = pd.DataFrame(data=probabilities)
df_probs.head()

