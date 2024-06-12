# disaster_tweets_nlp

## Project Objective
Build a machine learning model to predict which Tweets are about real disasters and which are not.

## Dataset Used

I downloaded the training dataset from ([Kaggle](https://www.kaggle.com/c/nlp-getting-started)). The dataset is in csv format and has 10,873 records. The training dataset was split into ‘train’, ‘validate’ and ‘test’ datasets to train the model and evaluate its performance. Within the training dataset, only ‘text’ and ‘target’ columns were relevant for model building.

## Approach
I adopted the following step-by-step approach:
- Data pre-processing – The ‘text’ column of the dataset was pre-processed by changing it to lower case, removing numbers, punctuation, stop words and other irrelevant terms. Also, lemmatization was done on it to reduce words to their base form.
- Data visualization – To understand data, visualization was done on it. Following insights were drawn:
    * About 43% of the tweets were classified as "Disaster," indicating no significant class imbalance.
    * Top bigrams for disaster tweets included terms like "suicide bomber," "northern California," and "oil spill," clearly indicating disaster-related content.
    * Non-disaster tweets tended to have slightly more words than disaster tweets.
    * Extracted five topics from the texts, revealing key topics such as California wildfire, oil spill, thunderstorm disaster, car/train crashes, and bomb attacks.
- XGBoost Model – After pre-processing the texts, the data was split into 80:20 ‘train’ and ‘test’ datasets. For XGBoost, the ‘text’ data was converted into a matrix TF-IDF features which were used as the input data for the model.
- Bidirectional LSTM Model – Glove embeddings with 200 dimensions were loaded. The pre-processed text data was tokenized and encoded into numerical representation through TextVectorization. The bidirectional LSTM model architecture was built, and the model was compiled using ‘adam’ optimizer and ‘AUC’ as the model evaluation metric.
- Bidirectional Encoder Representations from Transformers (BERT) Model – Since Bert can handle raw text with punctuation and stop words, I directly used the ‘text’ data. Bert tokenizer was used to split the text into tokens or individual words and to insert special tokens ([CLS] and [SEP]) The input sequences were padded to the maximum sentence length to ensure consistent length for all inputs. The resulting input sequences were split into 80:20 ‘train’ and ‘validate’ datasets. For updating the parameters during training, ‘AdamW’ optimizer was used. A dynamic learning rate was used using a Learning rate scheduler. “BertForSequenceClassification” model was leveraged for our classification problem. It includes a classification layer built upon the original Bert model. The model was trained using the ‘train’ dataset and the performance was evaluated on the ‘validate’ dataset.

## Evaluation Metric
The metric used for evaluating the models is AUC ROC score. The AUC score reflects the model's ability to distinguish between positive and negative cases, with higher scores signifying better predictive performance.

## Algorithms
XGBoost model is an ensemble machine learning model which combines predictions from multiple decision trees. Multiple decision trees are built in sequence such that the errors of the
current decision tree are used to improve the next decision tree. It can effectively handle complex datasets and achieve high predictive accuracy.

Bidirectional LSTM model is a type of recurrent neural network. It can extract information from input sequences from both directions. Conventional recurrent neural networks use only previous words to learn about the following words in a sequence. LSTM (Long-Short Term Memory) allows the model to learn long-term dependencies in a text meaning in a very long sentence, the model will be able to learn about the words appearing at the end of the sentence from the words at the beginning of the sentence.

Bert is a language model based on the transformer architecture. It includes multiple encoders with each encoder comprising multiple self-attention heads and feed-forward layers. Self-attention heads help the model to develop an encoding for each word by looking at other positions in the input sequence and pass the encodings to the feed-forward layer. These feed-forward layers use attention outputs to capture dependencies and semantic information within the textual data. Bert has been pre-trained on a large amount of text data enabling it to have rich textual information.

## Results
Bidirectional LSTM Model provided the best AUC Score (~85%) on the test dataset. Bert and XGBoost achieved AUC scores of ~82% and ~76% respectively.
