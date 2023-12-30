# Lab9 - Natural Language Processing

## NLP
 - Natural Language Processing(NLP) is the application of computational techniques to the analysis and synthesis of natural language and speech
 - Text analytics (also referred to as text mining) is an artificial intelligence (AI) technology that uses natural language processing(NLP) to transform the free (unstructured) text in documents and databases into normalized, structured data suitable for analysis.
 - Text mining is a process of explain sizeable textual data and find patterns

## NLP and Text Analysis
 - To put it simply, text analytics deals with the text itself, while NLP deals with the underlying metadata
 - For a sentence, "A major incarnation of the weather phenomenon known as EI Nino is gathering force in the Pacific Ocean"
 - Answering questions like - frequency counts of words, length of the sentence, presence/ absence of certain words etc. is text analysis
 - NLP on the other hand allows you to answer question like
   - What is the sentiment
   - What ate the keywords
   - What category of content it falls under
   - Which are the entities in the sentence...
 - NLP in most cases needs a good understanding of statistics and machine learning

## What Does NLP Do
 - is a branch of artificial intelligence that consists of systematic processes for analyzing, understanding, and deriving information from the text data in a smart and efficient manner
 - can organize the massive chunks of text data, perform numerous automated tasks and solve a wide range of problems such as - automatic summarization, machine translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, etc.

## NLTK
 - stands for Natural Language toolkit is open-source library in python
 - provide wide range of tools and resources for tasks.
 - Text Processing: NLTK offers various text processing capabilities, including tokenization (splitting text into words or sentences), stemming (reducing words to their base or root form), and lemmatization (reducing words to their dictionary form).
 - Part-of-Speech Tagging: NLTK provides pre-trained models and algorithms for part-of-speech (POS) tagging, which assigns grammatical tags to words in a sentence, such as noun, verb, adjective, etc.
 - Named Entity Recognition: NLTK includes tools for named entity recognition (NER), which identifies and extracts named entities like names, locations, organizations, and dates from text.
 - Parsing and Chunking: NLTK supports syntactic parsing and chunking, allowing you to analyze the grammatical structure of sentences and extract meaningful phrases or chunks.
 - Corpora and Lexical Resources: NLTK includes a wide range of corpora (large collections of text) and lexical resources, such as WordNet (a lexical database) and various language-specific datasets, which can be used for language analysis and experimentation.
 - Text Classification: NLTK offers functionalities for text classification tasks, allowing you to build and train classifiers for tasks like sentiment analysis, spam detection, topic categorization, and more.

## Text Preprocessing
 - Since, text is the most unstructured form of all the available data, various types of noise are present in it and the data is not readily analyzable without any pre-processing.
 - The entire process of cleaning and standardization of text, making it noise-free and ready for analysis is known as text preprocessing.
 - It is predominantly comprised of three steps:
   - Noise Removal
   - Lexicon Normalization
   - Object Standardization.

### Noise Removal
 - Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise.
 - For example – language stopwords (commonly used words of a language – is, am, the, of, in, etc.), URLs or links, punctuations and industry specific words.
 - This step deals with removal of all types of noisy entities present in the text.
 - A general approach for noise removal is to prepare a dictionary of noisy entities, and iterate the text object by tokens (or by words), eliminating those tokens which are present in the noise dictionary.

### Lexicon Normalization
 - Another type of textual noise is about the multiple representations exhibited by single word.
 - For example – “play”,  “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.
 - Though they mean different but contextually all are similar.
 - The step converts all the disparities of a word into their normalized form (also known as lemma).
 - Normalization is a pivotal step for feature engineering with text as it converts the high dimensional features (N different features) to the low dimensional space (1 feature), which is an ideal ask for any ML model.
 - The most common lexicon normalization practices are :
   - Stemming: Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s”, etc.) from a word.
   - Lemmatization: Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations).

### Object Standardization
 - Text data often contains words or phrases which are not present in any standard lexical dictionaries.
 - These pieces are not recognized by search engines and models.
 - Some of the examples are – acronyms, hashtags with attached words, and colloquial slangs.
 - With the help of regular expressions and manually prepared data dictionaries, this type of noise can be fixed, the code below uses a dictionary lookup method to replace social media slangs from a text

## Feature Engineering on Text
 - To analyze a preprocessed data, it needs to be converted into features.
 - Depending upon the usage, text features can be constructed using assorted techniques – Syntactical Parsing, Entities / N-grams / word-based features, Statistical features, and word embeddings.
 - Read on to understand these techniques in detail.

### Part of Speech Tagging
 - Apart from the grammar relations, every word in a sentence is also associated with a part of speech (POS) tag (nouns, verbs, adjectives, adverbs, etc.).
 - The POS tags defines the usage and function of a word in the sentence.

### N-Grams as Features
 - A combination of N words together are called N-Grams.
 - N grams (N > 1) are generally more informative as compared to words (unigrams) as features.
 - Also, bigrams (N = 2) are considered as the most important features of all the others.

### Statistical Features
 - Text data can also be quantified directly into numbers using several techniques.
 - Term Frequency – Inverse Document Frequency (TF-IDF) is a weighted model commonly used for information retrieval problems.
 - It aims to convert the text documents into vector models on the basis of occurrence of words in the documents without taking considering the exact ordering.
 - For Example – let say there is a dataset of N text documents, In any document “D”, TF and IDF will be defined as:
   - Term Frequency (TF) – TF for a term “t” is defined as the count of a term “t” in a document “D”
   - Inverse Document Frequency (IDF) – IDF for a term is defined as logarithm of ratio of total documents available in the corpus and number of documents containing the term T.

## Major NLP Tasks

### Sentiment Analysis
 - Sentiment analysis (also known as opinion mining) refers to the use of NLP, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.
 - Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.

### Text Classification
 - Text classification is one of the classical problem of NLP.
 - Notorious examples include – Email Spam Identification, topic classification of news, sentiment classification and organization of web pages by search engines.
 - Text classification, in common words is defined as a technique to systematically classify a text object (document or sentence) in one of the fixed category.
 - It is really helpful when the amount of data is too large, especially for organizing, information filtering, and storage purposes.
 - A typical natural language classifier consists of two parts: (a) Training (b) Prediction as shown in image on the next slide.
 - Firstly the text input is processes and features are created.
 - The machine learning models then learn these features and is used for predicting against the new text.
