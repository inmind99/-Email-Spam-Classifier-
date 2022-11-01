# Importing Relavant Libraries
import re
import nltk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Setting environment to ignore future warnings
import warnings
warnings.simplefilter('ignore')


# ## Reading Dataset
data = pd.read_csv(r"C:\Desktop\Project\Email Spam\data", sep="\t", names=["label", "mail"])
data.head()
data.shape

data.info()

plt.figure(figsize=(14, 7))
sns.countplot(data.label)
plt.show()

# Data Preprocessing
stemmer = nltk.stem.PorterStemmer()
clean_data = []
for i in range(len(data)):
    temp = re.sub('[^a-zA-Z]', ' ', data["mail"][i]) # Removing special words
    temp = temp.lower()
    temp = temp.split()
    temp = [stemmer.stem(word) for word in temp if word not in nltk.corpus.stopwords.words("english")] # Stemming words
    temp = ' '.join(temp)
    clean_data.append(temp)

# Feature Engineering
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)   # max_features argument getsimportant words repeated in most of mails.
x = cv.fit_transform(clean_data).toarray()

# Encoding the label column data
def encoder(x):
    if x=='ham':
        return 0
    else:
        return 1

y = list(map(encoder, data["label"]))

# Preparing Model
# Spliting into train & test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)

# Fitting Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report
result_table = confusion_matrix(y_test, y_pred)
print(result_table)

print(classification_report(y_test, y_pred))


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy of the Model is ",round(score*100, 2), "%")