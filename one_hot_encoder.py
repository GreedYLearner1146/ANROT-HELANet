# One hot encode all label training array.

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

##################### Meta-Training set ##############################

# One hot encode all label train array.

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(new_y_train)
#print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

##################### Meta-Valid set ##############################

# One hot encode all label valid array.

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# integer encode
label_encoder = LabelEncoder()
integer_encodedval = label_encoder.fit_transform(new_y_val)

# binary encode. As of New ver, use sparse_output instead of sparse.
onehot_encoderval = OneHotEncoder(sparse_output = False)
integer_encodedval= integer_encodedval.reshape(len(integer_encodedval), 1)
onehot_encodedval= onehot_encoderval.fit_transform(integer_encodedval)
