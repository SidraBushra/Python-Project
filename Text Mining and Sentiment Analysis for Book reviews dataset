import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#data load
BookData=pd.read_csv('books_data-1.csv')
BookRating=pd.read_csv('Books_rating.csv')
BookData.head()
BookRating.head()



#Id                     object
#Title                  object
#Price                 float64
#User_id                object
#profileName            object
#review/helpfulness     object
#review/score          float64
#review/time             int64
#review/summary         object
#review/text            object
#dtype: object




plt.hist(BookRating['review/score'])

(array([ 201688.,       0.,  151058.,       0.,       0.,  254295.,
              0.,  585616.,       0., 1807343.]),
 array([1. , 1.4, 1.8, 2.2, 2.6, 3. , 3.4, 3.8, 4.2, 4.6, 5. ]),
' <BarContainer object of 10 artists>')



plt.hist(BookRating['review/time'])

(array([2.10000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        2.19000e+02, 4.45924e+05, 8.26463e+05, 7.66338e+05, 9.61035e+05]),
 array([-1.00000000e+00,  1.36235519e+08,  2.72471039e+08,  4.08706559e+08,
         5.44942079e+08,  6.81177600e+08,  8.17413120e+08,  9.53648640e+08,
         1.08988416e+09,  1.22611968e+09,  1.36235520e+09]),
 "<BarContainer object of 10 artists>")

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(str(BookRating['review/text']))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fractions import Fraction

# In[162]:


# data load
BookData = pd.read_csv('books_data-1.csv')
BookRating = pd.read_csv('Books_rating.csv')
BookData.head()
BookRating.head()

# cleaning the imorted data:


# dropping NA values
BookRating = BookRating.dropna()
BookRating

# dropping 0 values
BookRating = BookRating.loc[(BookRating[:] != 0).all(axis=1)]
BookRating

# dropping 0/0 values
BookRating = BookRating.loc[(BookRating[:] != '0/0').all(axis=1)]
BookRating

# resetting index
BookRating = BookRating.reset_index(drop=True)
BookRating

# getting the length of the dataframe for the loop below
len(BookRating)

# Converting to float
result = []
for i in range(302953):
    a, b = BookRating['review/helpfulness'][i].split('/')
    result.append(float(a) / float(b))

# replacing the values in the dataframe with the above values
BookRating['review/helpfulness'] = result

plt.hist(BookRating['review/score'])

plt.hist(BookRating['review/time'])

x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)
word_cloud = WordCloud(collocations=False, background_color='white', stopwords=STOPWORDS, mask=mask).generate(
    str(BookRating['review/text']))
plt.imshow(word_cloud, interpolation='none')
plt.axis("off")
plt.show()

# The most common words are Seuss, book and Dr. and the review score is overwhelmingly high.

# Categorizing into positive those reviews with score of bigger than 2.5
PositiveSubset = BookRating.loc[(BookRating['review/score'] > 2.5)]
PositiveSubset.head()

# Categorizing into negative those reviews with score of less than 2.5
NegativeSubset = BookRating.loc[(BookRating['review/score'] < 2.5)]
NegativeSubset.head()

# positive word cloud
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)
word_cloud = WordCloud(collocations=False, background_color='white', stopwords=STOPWORDS, mask=mask).generate(
    str(PositiveSubset['review/text']))
plt.imshow(word_cloud, interpolation='none')
plt.axis("off")
plt.show()

# negative word cloud
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)
word_cloud = WordCloud(collocations=False, background_color='white', stopwords=STOPWORDS, mask=mask).generate(
    str(NegativeSubset['review/text']))
plt.imshow(word_cloud, interpolation='none')
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(BookRating.loc[:, BookRating.columns != 'review/score'],
                                                    BookRating['review/score'], test_size=0.33, random_state=42)

# logistic regression
logisticRegression = LogisticRegression()
logisticRegression.fit(np.array(X_train['review/helpfulness']).reshape(-1, 1), y_train)

logisticRegression.predict_proba(np.array(y_test).reshape(-1, 1))

# multinominal logistic regression

MultinomiallogisticRegression = LogisticRegression(multi_class='multinomial')
MultinomiallogisticRegression.fit(np.array(X_train['review/helpfulness']).reshape(-1, 1), y_train)

# prediction
MultinomiallogisticRegression.predict_proba(np.array(y_test).reshape(-1, 1))

