"""
In this example, we'll be using the relational database to create a standard ML problem.
The problem we'll be addressing is, how can we develop a model to predict worker satisfaction?
Worker satisfaction is a float value, and ranges between 0 and 1.

Notes: categorical data will be one-hot encoded.
Salary data is difference from median.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch.autograd import Variable
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

# set basepath
BasePath = os.path.dirname(__file__)

# set database path (for relational database)
RelationalPath = os.path.join(BasePath, 'data', 'relational-database.csv')

# import database with pandas
rel_data = pd.read_csv(RelationalPath)

# rename columns for readability
# Renaming certain columns for better readability
rel_data = rel_data.rename(columns={'satisfaction_level': 'satisfaction',
                                    'last_evaluation': 'evaluation',
                                    'number_project': 'projectCount',
                                    'average_montly_hours': 'averageMonthlyHours',
                                    'time_spend_company': 'yearsAtCompany',
                                    'Work_accident': 'workAccident',
                                    'promotion_last_5years': 'promotion',
                                    'sales': 'department',
                                    'left': 'turnover'
                                    })

# drop nans. Keep duplicates (no need to artificially influence data)
rel_data.dropna(inplace=True)

# one-hot encode the categorical features using pandas get_dummies
ohe_rel_data = pd.get_dummies(rel_data)

# for neural networks, make sure to normalize the non-categorical data
def normalize(df,non_categorical_columns):
    """ https://stackoverflow.com/a/26415620 """
    result = df.copy()
    for feature_name in non_categorical_columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

non_categorical_columns = ['satisfaction','evaluation','projectCount',
                           'averageMonthlyHours','yearsAtCompany']

ohe_rel_data = normalize(ohe_rel_data,non_categorical_columns)



# split the data into train_test (80%) and validate (20%) fractions
train_test, validate = np.split(ohe_rel_data.sample(frac=1),
                                [int(.8 * len(ohe_rel_data))])

# choose target feature; here we choose: 'satisfaction'
target_feature = 'satisfaction'

# extract it from the relational data
train_test_target_data = train_test[target_feature]
validate_target_data = validate[target_feature]

## store target it as an array for import into ML pipeline
train_test_target_data_numpy = np.array(train_test_target_data)
validate_target_data_numpy = np.array(validate_target_data)

# drop the target feature from the dataset (note: now we have 20 features)
train_test_training_data = train_test.drop([target_feature], axis=1)
validate_training_data = validate.drop([target_feature], axis=1)

# turn training data into a matrix
train_test_training_data_numpy = np.matrix(train_test_training_data)
validate_training_data_numpy = np.matrix(validate_training_data)

# examine the performance of the model with repeated
# cross validation loops of training data
# it's a regression problem, so no imbalance.
# However, instantiate a repeatedKfold
n_repeats = 1
repeatKfold = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)

# loop through the data considering the train/test splitting

for train_index, test_index in repeatKfold.split(train_test_training_data_numpy):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_test_training_data_numpy[train_index], train_test_training_data_numpy[test_index]
    y_train, y_test = train_test_target_data_numpy[train_index], train_test_target_data_numpy[test_index]

    # load the training and target data into tensors for pytorch
    ## if cuda is available, choose cuda tensors for speed
    ## make sure that you specify np.float32 values!
    ## also, you could do better if you normalized the vals...
    X_train = np.matrix(X_train, dtype=np.float32)
    y_train = np.matrix(y_train, dtype=np.float32)
    X_train = np.matrix(X_train, dtype=np.float32)
    y_train = np.matrix(y_train, dtype=np.float32)

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    if torch.cuda.is_available():
        X_train_tensor = torch.from_numpy(X_train).cuda()
        y_train_tensor = torch.from_numpy(y_train).cuda()
        X_test_tensor = torch.from_numpy(X_test).cuda()
        y_test_tensor = torch.from_numpy(y_test).cuda()

    # create variable wrap for autogradient
    ## if cuda is available, choose cuda tensors for speed
    X_train_tensor_Variable = Variable(X_train_tensor)
    y_train_tensor_Variable = Variable(y_train_tensor, requires_grad=False)
    X_test_tensor_Variable = Variable(X_test_tensor)
    y_test_tensor_Variable = Variable(y_test_tensor, requires_grad=False)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Variables for its weight and bias.

    d_in = X_train_tensor.shape[1]  # input dimension
    hidden_dim = 100  # hidden dimensions
    # the output dimension is the shape of the target vector
    # in this case, the shape is 1, because we are regressing a single value
    d_out = 1  # output dimension

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train_tensor.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, d_out), )

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(size_average=False)

    learning_rate = 1e-4
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_track = []
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Variable of input data to the Module and it produces
        # a Variable of output data.
        y_pred = model(X_train_tensor_Variable)

        # Compute and print loss. We pass Variables containing the predicted and true
        # values of y, and the loss function returns a Variable containing the loss.
        loss = loss_fn(y_pred, y_train_tensor_Variable)
        print(t, loss.data[0])
        loss_track.append(loss.data[0])

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    plt.plot(range(500), loss_track)

plt.show()
