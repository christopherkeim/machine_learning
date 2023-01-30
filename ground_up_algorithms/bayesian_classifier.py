#!/usr/bin/env python3
import math

class BayesianClassifier:
    """
    The BayesianClassifer base class.
    
    Note that this class does not have a classify() method.
    """
    def __init__(self, get_features):
        """
        Initialize the class with a get_features() function.
        """
        self.get_features = get_features
        
    def setdb(self, dbname):
        """
        This method opens a SQLite database for the classifer and creates the featureCounts and categoryCounts tables,
        if they don't exist yet.
        """
        self.con = sqlite.connect(dbname)
        self.con.execute("create table if not exists featureCounts(feature, category, count)")
        self.con.execute("create table if not exists categoryCounts(category, count)")
        
    def inc_feature_count(self, feature, category):
        """
        Database modifier method for the featureCounts table.
        
        Takes a feature and a category as inputs and increases the count value + 1 for that feature's occurence in the given
        category.
        """
        count = self.get_feature_count(feature, category)
        if count == 0:
            self.con.execute("insert into featureCounts values ('%s', '%s', 1)" % (feature, category))
        
        else:
            self.con.execute("update featureCounts set count=%d where feature='%s' and category='%s'" % (count+1, feature, category))
            
    def get_feature_count(self, feature, category):
        """Database access method for featureCounts table.
        
        Retrieves the count value for an input feature and category combination. Returns 0 if not in table.
        """
        result = self.con.execute("select count from featureCounts where feature='%s' and category='%s'" % (feature, category)).fetchone()
        if result == None:
            return 0
        
        else:
            return float(result[0])
            
    def inc_category_count(self, category):
        """
        Database modifier method for the categoryCounts table.
        
        Increases an input category's count + 1 or creates an entry if it doesn't exist yet.
        """
        count = self.get_catgory_count(category)
        if count == 0:
            self.con.execute("insert into categoryCounts values ('%s', 1)" % (category))
            
        else:
            self.con.execute("update categoryCounts set count=%d where category='%s'" % (count+1, category))
            
    def get_category_count(self, category):
        """
        Database access method for the categoryCounts table.
        
        Retrieves the count value for an input category or returns 0 if that category is not in the table.
        """
        result = self.con.execute("select count from categoryCounts where category='%s'" % (category)).fetchone()
        if result == None:
            return 0
        
        else:
            return int(result[0])
        
    def get_category_list(self):
        """
        Database access method for categoryCounts table.
        
        Constructs list of all categories stored in this table.
        """
        category_list = []
        current = self.con.execute("select category from categoryCounts")
        for c in current:
            category_list.append(c[0])
        return category_list
    
    def get_sample_total(self):
        """
        Database access method for categoryCounts table. 
        
        Returns the total number of samples that the classifier has been trained on so far, equal to category1 + category2 + 
        category3 + ...
        """
        result = self.con.execute("select sum(count) from categoryCounts").fetchone()
        if result == None:
            return 0
        
        return result[0]
    
    def train(self, sample, category):
        """
        The learning algorithm for the BayesianClassifer.
        
        Takes a data sample (here, a text document) and its category as inputs, breaks the sample into features, increases the
        the count value in the featureCounts table for every feature associated with this category, and then increases the
        count value in the categoryCount table for this category.
        """
        #Extract the features from this data sample.
        features = self.get_features(sample)
        #Update the featureCounts table for every feature in this sample.
        for f in features:
            self.inc_feature_count(f, category)
        #Update the categoryCounts table for this category.
        self.inc_category_count(category)
        #Save the changes from this incremental training session in the database.
        self.con.commit()
        
    def feature_probability(self, feature, category):
        """
        Calculates the conditional probability P(feature | category) by dividing self.get_feature_count() / self.get_category_count().
        """
        if self.get_category_count(category) == 0:
            return 0
        return self.get_feature_count(feature, category) / self.get_category_count(category)
    
    def weighted_probability(self, feature, category, prob_function, weight=1.0, assumed_prob=0.5):
        """
        Returns a weighted average of the probability function's output and the assumed probability.
        """
        #Calculate probability function's output for feature and category. 
        basic_prob = prob_function(feature, category)
        #The weight for this probability value is the total number of samples the feature has appeared in across all categories.
        cateogies = self.get_category_list()
        count = 0
        for c in categories:
            count += self.get_feature_count(feature, c)
        #Calculate the weighted average.
        wp = ((weight * assumed_prob) + (count * basic_prob)) / (weight + count)
        
        #Return the weighted probability. 
        return wp
        
class NaiveBayesClassifier(BayesianClassifer):
    """
    The NaiveBayesClassifer extends BayesianClassifer and adds functionalities specific to Naive Bayesian classification. 
    """
    def __init__(self, get_features):
        """
        The NaiveBayes class is also initialized with a get_features function, and sets up a thresholds dictionairy.
        """
        super().__init__(get_features)
        self.thresholds = {}
        
    def sample_probability(self, sample, category):
        """
        Calcultes the probability that a sample within a specific category (i.e. a "good" or "bad" documents, for example) contains all each feature
        it is composed of, P(sample | if category)
        """
        #Extract the features from the sample.
        features = self.get_features(sample)
        #Multiply the probabilites of all the features in the sample together.
        p = 1
        for f in features:
            p *= self.weighted_probability(f, category, self.feature_probability)
        return p
        
     def naive_probability(self, sample, category):
        """
        Calculates the Naive Bayes probability that a specific sample belongs to a specific category, P(category | sample), by multiplying the 
        sample_prob by the category_prob.
        """
        sample_prob = self.sample_probability(sample, category)
        category_prob = self.get_category_count(cateogry) / self.get_sample_total()
        return sample_prob * category_prob
    
    def set_threshold(self, category, t):
        """
        Setter method that takes a category and a threshold value and creates a 'category': threshold entry in the thresholds instance dictionary.
        """
        self.thresholds[category] = t
        
    def get_threshold(self, category):
        """
        Getter method that takes a category as input and returns its threshold value, or 1.0 if there is no threshold. 
        """
        if category not in self.thresholds:
            return 1.0
        return self.thresholds[category]
    
    def classify(self, sample, default=None):
        """
        Classification function for NaiveBayesClassifer. 
        
        Takes a new data sample as an input as well as a default value. Calculates the naive_probability for each category value, determines with probability
        is largest and whether this value exceeds the next largest value by the given threshold, and if so returns it. If not, it returns the passed in default value.
        """
        naive_probs = {}
        #Calculate naive probabilities for each category, find the highest value. 
        max = 0.0
        best = None
        for cat in self.get_category_list():
            naive_probs[cat] = self.naive_probability(sample, cat)
            if naive_probs[cat] > max:
                max = naive_probs[cat]
                best = cat
        #Make sure probability exceeds threshold * next best
        for c in naive_probs:
            if c == best:
                continue
            if naive_probs[c] * self.get_threshold(best) > naive_probs[best]:
                return default
        return best
       
class FisherBayesClassifier(BayesianClassifier):
    """
    The FisherBayesClassifier extends the BayesianClassifier class and adds functionalities specific to the Fisher technique for Bayesian classification.
    """
    def __init__(self, get_features):
        """
        The FisherBayesClassifier is also initialized with a get_features function and sets up a minumums dictionary. 
        """
        super().__init__(get_features)
        self.minimums = {}
        
    def clf_norm_probability(self, feature, category):
        """
        Calculates the normalized feature probability P(category | feature) by calculating P(feature | category) with the feature_probability() method, 
        creating a partition function Q = Σ P(feature | category) across all categories and returning the P(feature | category) / Q.
        """
        #Feature probability given a category.
        f_prob = self.feature_probability(feature, category)
        #Partition function is the sum of all probabilities for this feature across all categories.
        q = 0.0
        for c in self.get_category_list():
            q += self.feature_probability(feature, c)
        #Normalized probabilty P(cateogry | feature) is f_prob / q
        p = f_prob / q
        return p
    
    def fisher_probability(self, sample, category):
        """
        Calculates the resultant Fisher probability that a sample belongs in a category.
        
        Multiplies all of the clf_norm_probability() feature probabilities together, takes the nautral log of this product and multiplies it by -2 to 
        get the F-score, and inputs it into the inv_chi_2() inverse chi squared function.
        """
        #Extract all of the features from this sample.
        features = self.get_features(sample)
        #Multiply feature probabilities together.
        p = 1
        for f in features:
            p *= self.weighted_probability(f, category, self.clf_norm_probability)
        #Calculate F-score
        f_score = -2 * math.log(p)
        #Use the inverse chi squared function to get the Fisher probability.
        return self._inv_chi_2(f_score, len(features)*2)
    
    def inv_chi_2(self, chi, df):
        """
        The inverse chi squared function.
        """
        m = chi / 2.0
        term = math.exp(-m)
        sum = math.exp(-m)
        for i in range(1, df//2):
            term *= m / i
            sum += term
        return min(sum, 1.0)
    
    def set_minimum(self, category, min):
        """
        Setter method for the minimums instance dictionary. 
        
        Takes a category and a minimum value as inputs
        """
        self.minimums[category] = min
    
    def get_minimum(self, category):
        """
        Returns the minimum value for an input category or 0 if there is no set minimum. 
        """
        if category no in self.minimums:
            return 0
        return self.minimums[category]
    
    def classify(self, sample, default=None):
        """
        Classification method for the FisherBayesClassifier.
        
        Takes a new data sample as an input as well as a default value. Calculates the Fisher probability for each category, determines which probability
        is largest and if it exceeds the minimum value for the category, and if so it returns the category. If not, if not it returns the default value. 
        """
        max = 0.0
        best = None
        #Calculate the Fisher probabilities for each category, find the largest value.
        for c in self.get_catgory_list():
            p = self.fisher_probability(sample, c)
            #Make sure it's greater than the minimum for this category
            if p > self.get_minimum(c) and p > max:
                best = c
                max = p
        return best
               
    
        
        
        
        
        
        
