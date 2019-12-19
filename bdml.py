import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier as NN
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from tabulate import tabulate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# load the training/testing images and labels
trnImages = np.load('trnImage.npy')
trnClasses = np.load('trnLabel.npy')
trnidx = 20
tstImages = np.load('tstImage.npy')
tstClasses = np.load('tstLabel.npy')

plt.imshow(trnImages[:,:,:,trnidx])

def computeFeatures(imageDataSet):
    # This function computes the HOG features with the parsed hyperparameters and returns the features as hog_feature. 
    # By setting visualize=True we obtain an image, hog_as_image, which can be plotted for insight into extracted HOG features.
    hog_feature_list = []
    # Loops through every image in the dataset and computes its hog features
    for idx in range(imageDataSet.shape[-1]):
        image = imageDataSet[:,:,:,idx]
        hog_feature = skimage.feature.hog(image, block_norm='L2-Hys')
        hog_feature_list.append(hog_feature)
    return hog_feature_list

# Gets all labels in dataset
def getLabels(labels):
    return [label[0] for label in labels]

# Extract the features from a single image
#features, hog_image = computeFeatures(trnImages[:,:,:,trnidx])
training = computeFeatures(trnImages)
testing = computeFeatures(tstImages)
trainingLabels = getLabels(trnClasses)
testLabels = getLabels(tstClasses)
#plt.imshow(hog_image)
#plt.show(block=False)

#pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=324))])

# Transforms training and testing data sets using PCA
pcaModel = PCA().fit(training, trainingLabels)
pcaTraining = pcaModel.transform(training)
pcaTesting = pcaModel.transform(testing)

# Creates K-Means Model
kMeansModel = KMeans().fit(pcaTraining, trainingLabels)
plt.scatter(pcaTraining[:,0], pcaTraining[:,1])

# Creates GMM Model
gmmModel = GaussianMixture(n_components=10, covariance_type='diag').fit(pcaTraining, trainingLabels)

# Creates LDA Model
ldaModel = LDA().fit(training, trainingLabels)

# Creates LDA Model using PCA
ldaModelPca = LDA().fit(pcaTraining, trainingLabels)

# Creates SVM Model
svmModel = LinearSVC().fit(training, trainingLabels)

# Creates SVM Model using PCA
svmModelPca = LinearSVC().fit(pcaTraining, trainingLabels)

# Creates NN Model 
nnModel = NN(activation="identity").fit(training, trainingLabels)

#nnModelPca = NN(activation="identity", solver='adam').fit(pcaTraining, trainingLabels)

# Creates confusion matrix table
def evaluate(predictions, correct):
    tableCategories = {
        1: 'airplane',
        2: 'automobile',
        3: 'bird',
        4: 'cat',
        5: 'deer',
        6: 'dog',
        7: 'frog',
        8: 'horse',
        9: 'ship',
        10: 'truck'
    }
    comparison = {}
    for p,c in zip(predictions, correct):
        if p not in comparison:
            comparison[p] = {}
        if c not in comparison[p]:
            comparison[p][c]=0
        comparison[p][c]+=1
    headers = [tableCategories[idx] for idx in range (1,11)]
    table = []
    for pIdx in range(1,11):
        row = [tableCategories[pIdx]]
        if pIdx not in comparison:
            for blankIdx in range (1,11):
                row.append(0)
        else:
            for cIdx in range(1,11):
                if cIdx not in comparison[pIdx]:
                    row.append(0)
                else:
                    row.append(comparison[pIdx][cIdx])
        table.append(row)
    print(tabulate(table, headers, tablefmt="grid"))
    getProbability(comparison)
    
# Calculates probability that model predicts classification correctly
def getProbability(comparison):
    noOfCorrect = 0
    for idx in range(1,11):
        try:
            noOfCorrect += comparison[idx][idx]
        except:
            noOfCorrect += 0
    print("Percentage of correct values for all categories", noOfCorrect/10, "% \n")
        
# Displays all model predictions against actual results
for model in [kMeansModel, gmmModel, ldaModel, svmModel, nnModel]:
    print(model)
    # Using models to predict using original testing images
    predictions = model.predict(testing)
    
    #print(predictions)
    evaluate(predictions, testLabels)

# Displays all model predictions against actual results
for model in [kMeansModel, gmmModel, ldaModelPca, svmModelPca, nnModelPca]:
    print(model)
    # Using models to predict using tranformed testing images
    predictions = model.predict(pcaTesting)
    
    #print(predictions)
    evaluate(predictions, testLabels)

