# CP2
資料科學導論 CP2 
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("training_data.csv")
#print(df)
#df_one = df.loc[0:len(df)]
textlist_one = df["text"].tolist()

#print(textlist) testlist是1/10的text

#no801

#p36

S_words =["a", "about", "above", "above", "across", "after",
          "afterwards", "again", "against", "all", "almost", "alone", 
          "along", "already", "also","although","always","am","among",
          "amongst", "amoungst", "amount",  "an", "and", "another", "any",
          "anyhow","anyone","anything","anyway", "anywhere", "are", "around",
          "as",  "at", "back","be","became", "because","become","becomes", "becoming",
          "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
          "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can",
          "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", 
          "describe", "detail", "do", "done", "down", "due", "during", "each",
          "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even",
          "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", 
          "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front",
          "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her",
          "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
          "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", 
          "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
          "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover",
          "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", 
          "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
          "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one",
          "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out",
          "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
          "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", 
          "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
          "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", 
          "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
          "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though",
          "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward",
          "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via",
          "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
          "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
          "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
          "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words= S_words, min_df=7)


X1_train = vectorizer.fit_transform(textlist_one)
#x=X1_train.toarray()
#y=np.array(df_one["stars"].tolist())
y=df["stars"].values.tolist()

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=5,shuffle=True)
skf.get_n_splits(X1_train,y)
ans=[]

dfcp = pd.read_csv("test_data.csv")
X_cp_test = vectorizer.transform(dfcp["text"].tolist())


from sklearn.svm import SVC
model = SVC(kernel="linear")
ans=model.fit(X1_train,y).predict(X_cp_test)


    
Y_result = pd.Series(ans,index=dfcp["review_id"])
Y_result.to_csv("test_output(new1).csv")
