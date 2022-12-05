import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import random
import seaborn as sns

from nltk.corpus import stopwords
from nltk import FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def plot_word_frequency(words, top_n=125, corpus_name='Corpus'):
    """Function to plot the word frequencies"""
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    tempDict = {"Labels": labels, "Count": counts}
    plt.figure(figsize=(20,10))
    plt.title("Most Frequent Words in the " + corpus_name + " - Excluding STOPWORDS")
    plt.ylabel("Count")
    plt.xlabel("Word")
    plt.xticks(rotation=90)
    plot = sns.barplot(data= tempDict, x= "Labels", y= "Count")
    return plot



def lift_ratio(index : list, data : pd.DataFrame, capture_column : str):
    """Calculates the lift ratio from a given Pandas Dataframe for specified columns/features within the dataset

    Keyword arguments:
    index -- (list) object containing the items/things of interest to calculate the lift ratio on
    data -- (Pandas DataFrame) object with data that has feature/column names that are contained in the 'index' argument
    capture_column -- (variable) column to pull data from
    Return: Pandas DataFrame that displays the lift ratios for the given features
    """
    lift_dict = pd.DataFrame(index=index, columns=index)
    total_shape = data.shape[0]
    for i in range(len(index)):
        for j in range(len(index)):
            brand_1 = index[i]
            brand_2 = index[j]
            count_1 = 0
            count_2 = 0
            count_3 = 0
            
            for txt in data[capture_column].values:
                if brand_1 in txt and brand_2 in txt:
                    count_3 = count_3 + 1
                elif brand_1 in txt and brand_2 not in txt:
                    count_1 = count_1 + 1 
                elif brand_1 not in txt and brand_2 in txt:
                    count_2 = count_2 + 1
            
            if(brand_1==brand_2):
                ans = np.nan
            else:
                pa = count_1/total_shape
                pb = count_2/total_shape
                pab = count_3/total_shape
                try:
                    ans = (pa*pb)/pab
                except ZeroDivisionError:
                    ans = 0
                
            if ans == 0:
                lift_dict.iat[i, j] = 0 #[brand_1][brand_2] = 0
            else:
                lift_dict.iat[i, j] = round(1/ans, 3) #[brand_1][brand_2] = round(1/ans,3)
    return lift_dict.apply(pd.to_numeric).style.background_gradient(axis=0,cmap='Blues'), lift_dict




def multi_dimensional_scaling(mdslifts : pd.DataFrame, points : list):
    """Display a multi-dimensional scaling graph

    Keyword arguments:
    mdslifts -- (Pandas DataFrame) object that contains the lift values for features 
    points -- (list) that contains the desired feature names found in the 'mdslifts' (Pandas DataFrame)
    Return: Displays a multi-dimensional scaling graph of the given dataframe's features
    """
    mdslifts.reset_index(drop=True, inplace=True)
    mds = MDS(random_state=0)
    mdslifts = mds.fit_transform(mdslifts)

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(points))]

    size = [64 for i in range(len(points))] ##41FB29
    fig = plt.figure(2, (10,4))
    plt.scatter(mdslifts[:,0], mdslifts[:,1], s=size, c=colors)
    count = 0
    for x, y in zip(mdslifts[:,0], mdslifts[:,1]):
        label = "{0}".format(points[count])
        count += 1
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha=('center') # horizontal alignment
        )
    plt.title('mds')
    return plt.show()


# NOTE: Can improve this code by making it able to accept multiple arguments ex. using **args with nested loop of comparison
# What I mean is take the "If value > threshold: ..." and make it its own function to accept any number of keywords.
def lift_adjustment(df : pd.DataFrame, threshold_1 : float, threshold_2 : float, threshold_3 : float, scale_1 : float, scale_2 : float, scale_3 : float):
    """Applies scalar modification to lift value dataframe to reshape for MDS plot

    Keyword arguments:
    df -- (Pandas Dataframe)
    threshold_1 -- (float) value determine which lift value to modify
    threshold_2 -- (float) value determine which lift value to modify
    scale_1 -- (float) value to modify lift value
    scale_2 -- (float) value to modify lift value
    Return: Dataframe with the rescaled/modified lift values
    """
    col_names = [x for x in df.columns]
    new_arr = []
    for row in df.values:
        new_val = []
        for value in row:
            if value > threshold_1:
                value = value * scale_1
            if value > threshold_2:
                value = value * scale_2
            if value > threshold_3:
                value = value * scale_3
            new_val.append(value)
        new_arr.append(new_val)
    complete_df = pd.DataFrame(new_arr, columns=col_names)
    return complete_df

def build_corpus(text_col):
    """To build a text corpus by stitching all the records together.Input the text column"""
    corpus = ""
    for word in text_col:
        corpus += word
    return corpus

"""_summary_
"""
def sentiment(series_text : pd.Series, series_topics : pd.Series):
    col_names = ["Content", "Word", "Negative", "Neutral", "Positive", "Compound"]
    topics = [str(topic).lower() for topic in series_topics.values]
    data = np.array(col_names)
    analyzer = SentimentIntensityAnalyzer()
    series_text = series_text.apply(lambda x: x.lower())
    series_text = series_text.apply(lambda x: x.split())
    for txt in series_text.values:
        corpus = build_corpus(txt)
        for topic in topics:
            if topic in txt:
                vs = analyzer.polarity_scores(build_corpus(corpus))
                array = np.array(
                    [corpus, topic, vs["neg"], vs["neu"], vs["pos"], vs["compound"]]
                )
            else:
                array = np.array(
                    [corpus, topic, pd.NaT, pd.NaT, pd.NaT, pd.NaT]
                )
            data = np.row_stack((data,array))
    
    data = pd.DataFrame(data=data, columns=col_names)
    data.drop(index=0, inplace=True)
    return data


"""_summary_
"""
def sentimentDialogue(series : pd.Series):
    analyzer = SentimentIntensityAnalyzer()
    headers = ["Post", "Negative", "Neutral", "Positive", "Compound"]
    posts_sentiments = np.array([headers])
    for post in series.apply(lambda x: x.lower()):
        vs = analyzer.polarity_scores(post)
        senti = np.array([
            post,
            vs['neg'],
            vs['neu'],
            vs['pos'],
            vs['compound']
        ])
        posts_sentiments = np.row_stack((posts_sentiments, senti))
    posts_sentiments = pd.DataFrame(data=posts_sentiments, columns=headers)
    posts_sentiments.drop(axis=0, index=0, inplace=True)
    avg_neg = np.average(posts_sentiments['Negative'])
    avg_neu = np.average(posts_sentiments['Neutral'])
    avg_pos = np.average(posts_sentiments['Positive'])
    avg_comp = np.average(posts_sentiments['Compound'])
    return posts_sentiments, avg_neg, avg_neu, avg_pos, avg_comp

"""_summary_
"""
def plot_sentiments(posts_sentiments : pd.DataFrame):
    # Visuallying the sentiments
        # Reformatting for clarity
    neg = posts_sentiments['Negative']
    neu = posts_sentiments['Neutral']
    pos = posts_sentiments['Positive']

        # Creating the plot
    fig_senti = plt.figure((20, 16))
    ax = plt.axes(projection='3d')
    m = ['o', '^', 's']
    colors = ["Red", "Blue", "Green"]

    ax.scatter(neg, neu, pos, c=colors, marker=m)

    # Add graph info
    ax.legend(loc='best')
    ax.set_title("Cluster of Sentiment Scores")
    ax.set_xlabel('Negative')
    ax.set_ylabel('Neutral')
    ax.set_zlabel('Positive')

    # Display the figure
    plt.show()

def plot_sentiment_groups(convo_sentiments: np.array, resp_sentiments : np.array):
    """ Creates a bar chart grouping the bot's and user sentiments to show visible difference.

    Arguments:
        convo_sentiments {numpy array} -- array of negative, neutral, positive, and compound scores (in that order) from the bot.
        resp_sentiments {numpy array} -- (numpy array) array of negative, neutral, positive, and compound scores (in that order) from the user.
    Returns: 
        None
    """
    labels = ['Negative', 'Neutral', 'Positive', 'Compound']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplot()
    rects1 = ax.bar(x - width/2, convo_sentiments, width, label='Bot\'s')
    rects2 = ax.bar(x + width/2, resp_sentiments, width, label='User')

    ax.set_ylabel('Percentage')
    ax.set_title('Exchange Sentiments by Bot and User')
    ax.set_xticks(x, labels)
    ax.legend(loc='best')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

def plot_sentiment_over_exchange(convo_senti : pd.DataFrame, resp_senti : pd.DataFrame, time = None, subSentiments = False):
    """Generates a plot that matches the change in sentiment over the span of the conversation.

    Arguments:
        convo_senti {pandas Dataframe} -- Negative, neutral, positive, and compound sentiment scores for the bot.
        resp_senti {pandas Dataframe} -- Negative, neutral, positive, and compound sentiment scores for the user.
        time {dateTime} -- Timestamps for the exchanges between the bot and the user (default: {None}).
        subSentiments {bool} -- Indicate whether to plot Negative, neutral, and positive sentiment scores (default: {False}).
    Return:
        None
    """

    # Create the difference for across each exchange
    convo_senti = convo_senti.apply(lambda x: x + 1)
    resp_senti = resp_senti.apply(lambda x : x + 1)
    diff_senti = convo_senti - resp_senti

    # Construct the labels for the lines
    labels = ['Negative', 'Neutral', 'Positive', 'Compound']

    # Create the x-axis, time, to track exchanges over
    if time is None:
        time = np.arange(1, diff_senti.shape[0])
    
    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()

    # Plot data based on passed booleans
    if subSentiments is False:
        plt.plot(time, diff_senti[labels[3]], fmt='ro', label=labels[3])
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Difference in Compound score')
    elif subSentiments is True:
        markers = ['v', 'o', '^', 's']
        colors = ['r', 'k', 'g', 'm']
        package = []
        for l, m, c in [(labels[0], markers[0], colors[0]), (labels[1], markers[1], colors[1]), (labels[2], markers[2], colors[2]), (labels[3], markers[3], colors[3])]:
            plt.subplot()
            plt.plot(time, diff_senti[labels[0]], label=l, marker=m, color=c)
    plt.show()