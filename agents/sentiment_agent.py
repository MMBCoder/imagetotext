from textblob import TextBlob


def analyze_sentiment(company):
    sample_reviews = [
        'Cadbury chocolate tastes amazing',
        'Packaging quality has reduced recently',
        'Excellent festive gifting option',
    ]

    sentiments = []

    for review in sample_reviews:
        polarity = TextBlob(review).sentiment.polarity
        sentiments.append({
            'review': review,
            'polarity': polarity,
        })

    return sentiments
