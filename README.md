# Fake News Detector

A fake news classifier built with scikit-learn, TF-IDF, and a passive-aggressive classifier. It gets 98.57% accuracy on the standard benchmark dataset and 76.67% on real, current news pulled live from an API. That gap is the actual point of this project.

## What it does

Trains a text classifier to label news articles as real or fake, using TF-IDF features and an online learning algorithm (passive-aggressive / SGD hinge-loss). Standard stuff. What makes this project worth looking at isn't the pipeline, it's what happened when I tested the model on news it had never seen before.

## The data

Two sources, used for two different purposes:

- **Kaggle's "Fake and Real News Dataset"** (Clément Bisaillon, based on the ISOT dataset), ~23K fake and ~21K real articles from 2016-2017. This is what the model trains and tests on.
- **A live pull from News API** (30 current real articles from BBC, Reuters, AP, and CNN). This isn't part of training at all. It exists purely as a holdout check: can the model recognize real news it's never structurally seen before, or did it just memorize quirks of one dataset?

That second question turned out to matter a lot.

## Results

| Model | Kaggle test accuracy |
|---|---|
| Majority-class baseline | 54.21% |
| Naive Bayes | 93.54% |
| Logistic regression | 98.30% |
| Passive-aggressive classifier | 98.57% |

On paper, 98.57% looks like a finished project. It isn't. Here's why.

## The leak

My first trained model scored 99.19% on the Kaggle test set. Great number. Then I ran it against the 30 fresh, real News API articles, ones it should have called "real" every single time, and it got **15 out of 30**. Fifty percent. Coin-flip odds on a binary classification problem.

I pulled the model's feature coefficients to see what it was actually keying on:

```
Strongest words pushing toward REAL:
  reuters      28.419
  wednesday     5.169
  washington    5.061
  ...
```

`reuters` alone had a coefficient more than five times larger than the next word. Turns out this is a known problem with this exact dataset: nearly every "real" article in it was scraped from Reuters.com, and almost all of them literally begin with a dateline like "WASHINGTON (Reuters) —". The model wasn't learning to detect fake news. It was learning to detect the word "Reuters."

I stripped that word out and retrained. Kaggle accuracy dropped slightly to 98.57% (expected, and honestly a good sign, it meant the shortcut was gone). The News API holdout jumped to 76.67%.

Better. Not fixed. I checked the coefficients again:

```
Strongest words pushing toward REAL:
  washington    7.884
  wednesday     7.290
  thursday      6.092
  ...
```

Removing "reuters" didn't remove the underlying pattern, it just pushed the model onto the next-closest proxy: city names, weekday mentions, the words "spokesman" and "statement." The whole wire-service writing style was the actual signal all along, "reuters" was just one visible piece of it. Chasing individual words after this point is a whack-a-mole game with no clean endpoint.

The 7 articles it still gets wrong on the holdout set make the pattern obvious: celebrity coverage, feature pieces, listicle-style headlines. Nothing in that style resembles a Reuters dispatch, so the model has nothing to grab onto and defaults to "fake."

## So what does this actually tell you

That a model can post a 98%+ accuracy number and still have learned almost nothing about what makes news fake. It learned to recognize a specific publisher's house style. On the exact kind of held-out test split most tutorials stop at, that model looks excellent. Point it at anything written differently and it falls apart.

I think this is a more useful result than a clean 99% would have been. A lot of published fake-news-detection demos use this same dataset without ever checking for this, which means a lot of reported accuracy numbers in this space are probably some flavor of the same problem.

## What I'd still want to fix

- The fake-article side has its own leftover artifacts (`getty`, `wire`, `image`, `video`) from whatever scraper built that half of the dataset. I didn't touch these, since they don't affect the real-news holdout test, but they're the same category of problem on the other side.
- The dataset is entirely 2016-2017 US politics. Words like `hillary` and `gop` are doing real work in the fake-word list, which won't generalize to news about anything else, any other era, or any other country.
- 30 holdout articles is a small sample. Directionally informative, not statistically bulletproof.
- A cleaner fix than word-stripping would be training on a dataset that doesn't have a single-source leak baked in, or deliberately removing datelines/bylines before any word ever reaches the vectorizer.

## Project structure

```
fake-news-detector/
├── data/
│   ├── raw/              # Fake.csv, True.csv, news_api_real.csv
│   └── processed/        # kaggle_combined.csv, train.csv, test.csv
├── src/
│   ├── fetch_news_api.py     # pulls the live holdout set
│   ├── prepare_data.py       # merges and labels the Kaggle files
│   ├── text_utils.py         # shared text cleaning, incl. artifact stripping
│   ├── preprocess.py         # cleans text, does the train/test split
│   ├── baseline.py           # majority-class, Naive Bayes, logistic regression
│   ├── train.py              # PassiveAggressiveClassifier (original API)
│   ├── train_sgd.py          # SGDClassifier equivalent, future-proof version
│   ├── evaluate.py           # accuracy, confusion matrix, classification report
│   ├── evaluate_holdout.py   # tests against the live News API pull
│   └── inspect_features.py   # prints top TF-IDF coefficients per class
├── models/                # saved model + vectorizer (joblib)
├── reports/               # classification_report.txt, confusion_matrix.png
└── requirements.txt
```

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll need two credentials, neither goes in git:

- A News API key in `.env` as `NEWS_API_KEY=...` (free tier at newsapi.org)
- A Kaggle API token at `~/.kaggle/access_token` (from kaggle.com/settings/api)

Then, in order:

```bash
python3 src/fetch_news_api.py
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
python3 src/prepare_data.py
python3 src/preprocess.py
python3 src/baseline.py
python3 src/train.py          # or train_sgd.py
python3 src/evaluate.py
python3 src/evaluate_holdout.py
python3 src/inspect_features.py
```

## A note on `PassiveAggressiveClassifier`

scikit-learn deprecated this class in version 1.8 and removes it entirely in 1.10, folding it into `SGDClassifier`. Both `train.py` (original) and `train_sgd.py` (the replacement, using `SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0)`) are included, and both produce matching results. If you're on scikit-learn 1.10+, `train.py` will throw an error since the class no longer exists, use `train_sgd.py`, or pin `scikit-learn<1.10` in `requirements.txt`.

## Data sources

- [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon, based on the ISOT Fake News Dataset (Ahmed et al.)
- [NewsAPI.org](https://newsapi.org) for the live holdout articles
