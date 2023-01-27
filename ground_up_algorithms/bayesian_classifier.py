#!/usr/bin/env python3
import math
import re

class BayesianClassifier:
    """
    The BayesianClassifer base class.
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
            
    def inc_category(self, category):
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
        
     
        
        
        
        
        
        
