import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao
from sklearn.linear_model import LinearRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0
     
    feat = my_map(X_train)

    y = 2 * y_train - 1

    model = LinearSVC(C=1, max_iter=50,tol=1e-2)
    model.fit(feat, y)

    w = model.coef_.flatten()
    b = model.intercept_[0]
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    F = 1 - 2 * X

    N, n = F.shape


    first_order = np.hstack([np.ones((N, 1)), F])

    second_order = []
    for i in range(n):
        for j in range(i+1, n):
            second_order.append(F[:, i] * F[:, j])
    second_order = np.stack(second_order, axis=1)

    '''third_order = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                third_order.append(F[:, i] * F[:, j] * F[:, k])
    third_order = np.stack(third_order, axis=1) '''

    high_order_chain = np.ones((N, n)) 
    for i in range(n):
        high_order_chain[:, i] = np.prod(F[:, i:], axis=1)

    high_order_outer = []

    for i in range(n):
      for j in range(i+1,n):
         high_order_outer.append(high_order_chain[:, i] * high_order_chain[:, j])

    high_order_outer = np.stack(high_order_outer, axis=1)

    feat = np.hstack([first_order, second_order, high_order_chain, high_order_outer])

    var = np.var(feat, axis=0)
    keep_var = var > 1e-5
    feat = feat[:, keep_var]

    return feat
    

################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors
	   
    p = np.zeros(64)
    q = np.zeros(64)
    s = np.zeros(64)
    r = np.concatenate(([2 * w[0]], 2 * w[1:64]))
    r = np.cumsum(r)
    p[63]=w[63]+w[64]+r[62]/2
    r[63]=w[63]-w[64]+r[62]/2
    tmp = abs(min(min(r),p[63]))
    p += tmp
    q += tmp
    r += tmp
    s += tmp
    return p, q, r, s

