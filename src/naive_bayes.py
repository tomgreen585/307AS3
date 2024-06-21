import sys
import pandas as pd

class naive_bayes:
    #init function for vars
    def __init__(self):    
        self.c = [] # set of all classes
        self.c_count = {} # class -> count(y)
        self.f_count = {} # feature -> count(Xi, xi, y)
        self.f_total = {} # feature -> total(Xi, y)
        self.c_prob = {} # class -> prob(y)
        self.f_prob = {} # feature -> prob(Xi, xi, y)
     
    #Training of Naive Bayes Classifier
    #Input: The training set -> classes, features, feature_name
    #Output: A probability table      
    def training(self, classes, feature, feature_name):
        self.c = set(classes)
        printfile = open("nb_train_output.txt", "w")
        
        #Initialise the count numbers to 1    
        for c_label in self.c: #for each class label y do
            self.c_count[c_label] = 1 #count(y) = 1
            for X_i in range(len(feature)): #for each feature Xi do
                for x_i in range(len(feature[X_i])): #for each possible value xi of feature Xi do
                    if feature_name[x_i] not in self.f_count:
                        self.f_count[feature_name[x_i]] = {} #counts(Xi) = 0
                    if c_label not in self.f_count[feature_name[x_i]]:
                        self.f_count[feature_name[x_i]][c_label] = {} #counts(Xi, xi) = 0
                    if feature[X_i][x_i] not in self.f_count[feature_name[x_i]][c_label]:
                        self.f_count[feature_name[x_i]][c_label][feature[X_i][x_i]] = 1 #counts(Xi, xi, y) = 1
        
        #Count the numbers of each class and feature value based on the training instances           
        for i in range(len(feature)): #for each training instance i [X1 = xi,...,Xn = xn, Y = y] do
            self.c_count[classes[i]] += 1 #count(y) += count(y) + 1
            for X_i in range(len(feature[i])): #for each feature Xi do
                if feature[i][X_i] not in self.f_count[feature_name[X_i]][classes[i]]:
                    self.f_count[feature_name[X_i]][classes[i]][feature[i][X_i]] = 1
                else: #count(Xi, xi, y) += count(Xi, xi, y) + 1
                    self.f_count[feature_name[X_i]][classes[i]][feature[i][X_i]] += 1
        
        #Calculate the total/denominators
        class_total = 0 #class_total = 0
        for c_label in self.c: #for each class label y do
            class_total += self.c_count[c_label] #class_total = class_total + count(y)
            for X_i in range(len(feature[i])): #for each feature Xi do
                if feature_name[X_i] not in self.f_total: 
                    self.f_total[feature_name[X_i]] = {} #total(Xi) = 0
                if c_label not in self.f_total[feature_name[X_i]]:
                    self.f_total[feature_name[X_i]][c_label] = 0 #total(Xi, y) = 0
                for counted in self.f_count[feature_name[X_i]][c_label]:
                    #total(Xi, y) = total(Xi, y) + count(Xi, xi, y)
                    self.f_total[feature_name[X_i]][c_label] += self.f_count[feature_name[X_i]][c_label][counted]
                    
        #Calculate the probabilities from the counting numbers
        for c_label in self.c: #for each class label y do
            self.c_prob[c_label] = self.c_count[c_label] / class_total #prob(y) = count(y) / class_total
            print("Train Class = " + c_label + ", Probability = " + str(self.c_prob[c_label])) #print class probs
            for X_i in range(len(feature[i])): #for each feature Xi do
                if feature_name[X_i] not in self.f_prob:
                    self.f_prob[feature_name[X_i]] = {} #prob(Xi) = 0
                if c_label not in self.f_prob[feature_name[X_i]]:
                    self.f_prob[feature_name[X_i]][c_label] = {} #prob(Xi, xi) = 0
                for counted in self.f_count[feature_name[X_i]][c_label]:
                    class_feature = self.f_count[feature_name[X_i]][c_label][counted] #count(Xi, xi, y)
                    total_feature = self.f_total[feature_name[X_i]][c_label] #total(Xi, y)
                    self.f_prob[feature_name[X_i]][c_label][counted] = class_feature / total_feature #prob(Xi, xi, y) = count(Xi, xi, y) / total(Xi, y)
        
        #Output the probabilities to file
        for c_label in self.c: #for each class label y do
            printfile.write("P(" + c_label + ") = " + str(self.c_prob[c_label]) + "\n")
            for X_i in range(len(feature[i])): #for each feature Xi do
                for counted in self.f_count[feature_name[X_i]][c_label]:
                    prob_feature = self.f_prob[feature_name[X_i]][c_label][counted]
                    printfile.write("P(" + feature_name[X_i] + " = " + str(counted) + " | " + str(c_label) + ") = " + str(prob_feature) + "\n")
        
        #return probability table -> class and feature probs              
        return self.c_prob, self.f_prob #return prob

    #Calculation of the class score
    #Input: A test instance [X1 = x1,...,Xn = xn], a class label y, the probability table prob
    #Output: The score
    def score(self, test_features, test_feature_name, p_class, p_feature, printfile):
        score = {} #score = 0
        
        for c_label in p_class: #for each class label y do
            score[c_label] = p_class[c_label] #score = prob(y)
            for X_i in range(len(test_features)): 
                for x_i in range(len(test_features[X_i])): #for each feature Xi do
                    score[c_label] *= p_feature[test_feature_name[x_i]][c_label][test_features[X_i][x_i]]#score = score * prob(Xi, xi, y)
            print("score(" + str(c_label) + " = " + str(score[c_label]) + ")")
            printfile.write("score(" + str(c_label) + " = " + str(score[c_label]) + ")\n")
            
        return score #return score

# Prediction of the test set
def predict_test(test_classes, test_features, test_feature_name, p_class, p_feature):
    printfile = open("nb_test_output.txt", "w")
    class_prediction = []
    test_itr = 1  #test itr number
    correct = 0#correct num of predictions
    
    for features in test_features:  # for each test instance [X1 = xi,...,Xn = xn, Y = y] do
        print("\nTest Instance = " + str(test_itr))
        printfile.write("\nTest Instance = " + str(test_itr) + "\n")
        score = n.score([features], test_feature_name, p_class, p_feature, printfile)
        s = max(score, key=score.get) #get max score
        class_prediction.append(s) #append score to class_prediction
        test_itr += 1  #increment test itr number
    
    for i in range(len(class_prediction)): #for index[i] in range predicted_classes
        predicted = class_prediction[i] #get predicted class -> index[i]
        actual = test_classes[i] #get actual class -> index[i]
        if predicted == actual: #if predicted class = actual class
            correct += 1 #correct = correct + 1
        print("\nPredicted Test Class " + str(i+1) + " = " + predicted + ", Actual Test Class = " + actual)
        printfile.write("\nPredicted Test Class " + str(i+1) + " = " + predicted + ", Actual Test Class = " + actual)
    accuracy = correct / len(test_classes) #accuracy = correct / total
    print("ACCURACY = " + str(accuracy * 100) + "%")
    printfile.write("\nACCURACY = " + str(accuracy * 100) + "%")

#main
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python3 naive_bayes.py breast-cancer-training.csv breast-cancer-test.csv")
        sys.exit(1)
    
    #get training data breast-cancer-training.csv
    train = pd.read_csv(sys.argv[1], delimiter=',')
    classes = train.iloc[:, 1].tolist()
    features = train.iloc[:, 2:].values.tolist()
    feature_name = train.columns[2:].tolist()
    
    n = naive_bayes()#initialise naive_bayes
    p_class, p_feature = n.training(classes, features, feature_name) #call training -> get prob table (class, feature)
    
    #get test data breast-cancer-test.csv
    test = pd.read_csv(sys.argv[2], delimiter=',')
    test_classes = test.iloc[:, 1].tolist()
    test_features = test.iloc[:, 2:].values.tolist()
    test_feature_name = test.columns[2:].tolist()
    
    #call predict_test -> get predicted classes
    predicted_test = predict_test(test_classes, test_features, test_feature_name, p_class, p_feature)