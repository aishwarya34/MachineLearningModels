from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()
# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image data shape", digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label data Shape", digits.target.shape)

# Splitting Data into Training and Test Sets (Digits Dataset)
x_train,x_test,y_train,y_test= train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)

# Scikit-learn Modeling Pattern
# # all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

#Training the model on the data, storing the information learned from the data
logisticRegr.fit(x_train,y_train)

#Predict labels for new data (new images)

# Returns a NumPy Array
# Predict for One Observation (image)
# p = logisticRegr.predict(x_test[0].reshape(1,-1))
# print("First predict: " ,p)


#Make predictions on entire test data
predictions = logisticRegr.predict(x_test)

#Measuring Model Performance (Digits Dataset)
# accuracy is defined as:
# (fraction of correct predictions): correct predictions / total number of data points
score = logisticRegr.score(x_test,y_test)
print(score)


'''train_test_split(..,random_state=0) splits arrays or matrices into random train and test subsets. That means that everytime you run it without specifying random_state, you will get a different result, this is expected behavior. 
                                        On the other hand if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same. 
                                        It doesn't matter what the actual random_state number is 42, 0, 21, ... The important thing is that everytime you use 42, you will always get the same output the first time you make the split.
                                         This is useful if you want reproducible results, for example in the documentation, so that everybody can consistently see the same numbers when they run the examples. In practice I would say, 
                                         you should set the random_state to some fixed number while you test stuff, but then remove it in production if you really need a random (and not a fixed) split.'''