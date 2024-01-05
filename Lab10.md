# Lab10 - Introduction to Recommender Systems

## Recommender System
 - A recommender system is a type of information filtering system that suggests relevant items to users based on their preferences, interests, and behavior. It is commonly used in e-commerce, social media platforms, movie and music streaming services, and other online platforms to provide personalized recommendations to users.
 - It uses various algorithms and techniques, such as matrix factorization, nearest neighbour algorithms, and machine learning models to generate recommendations.
 - Examples like e-commerce, people usually purchase products based on the reviews given by relatives or friends but now as the options increased and we can buy anything digitally we need to assure people that the product is good and they will like it.
 - Examples: Homepage Recommendations
   - Homepage recommendations are personalized to a user based on their known interests. Every user sees different recommendations.
   - Examples: Google Play Store Apps
 - Example: Related Item Recommendations
   - related items are recommendations similar to a particular item
   - In the Google Play apps example, users looking at a page for a math app may also see a panel of related apps, such as other math or science apps.

## Terminology
 - Items – entities a system recommends
 - Query – information a system uses to make recommendations
   - User information: uid / browse record
   - Additional Content: time / user’s device
 - Embedding – A mapping from a discrete set

## Recommendation Systems
 - Common architecture for recommendation system consists of the following components:
   - Candidate Generation
   - Scoring
   - Re-ranking

## Components
 - Candidate Generations: This method is responsible for generating smaller subsets of candidates to recommend to a user, given a huge pool of thousands of items.
 - Scoring Systems: Candidate Generations can be done by different Generators, so, we need to standardize everything and try to assign a score to each of the items in the subsets. 
 - Re-Ranking Systems: After the scoring is done, along with it the system takes into account other additional constraints to produce the final rankings.

## Candidate Generation
 - Content-based filtering System
 - Collaborative filtering System

### Content-based filtering system
 - This approach recommends items similar to those that a user has liked or interacted with in the past. 
 - It utilizes item metadata, such as genre, keywords, or descriptions, to find items with similar attributes.

#### User Interaction
 - Recommender systems require certain feedback to perform recommendations.
   - Information on users’ past behaviour, the behaviour of other people, or the content information of the domain to produce predictions

#### Feedback
 - Implicit Feedback: The user’s likes and dislikes are noted and recorded on the basis of his/her actions like clicks, searches, and purchases. They are found in abundance but negative feedback is not found.
 - Explicit Feedback: The user specifies his/her likes or dislikes by actions like reacting to an item or rating it. It has both positive and negative feedback but less in number

#### Similarity Measures
 - During recommendation, the similarity metrics are calculated from the item’s feature vectors and the user’s preferred feature vectors from his/her previous records.
 - To determine the degree of similarity, most recommendation systems rely on one or more of the following:
   - cosine
   - dot product
   - Euclidean distance
 - We will not discuss the details of similarity measures in this lecture.

#### Advantages & Disadvantages
 - Advantages
   - Specific to a dedicated user, doesn’t need any data about other users.
   - Capture the specific interests of a user.
 - Disadvantages
   - Requires a lot of domain knowledge.
   - Can only make recommendations based on existing interests of user.

### Collaborative filtering System
 - This approach recommends items based on the preferences of similar users. It looks for patterns and similarities in user behavior or item ratings to make recommendations.
 - There are two categories of collaborative filtering:
   - User-Based Collaborative Filtering
   - Item-based Collaborative Filtering
 - a technique used to predict the items that a user might like on the basis of ratings given to that item by other users who have similar taste with that of the target user.
 - Many websites use collaborative filtering for building their recommendation system.
 - a technique to predict a user’s taste and find the items that a user might prefer on the basis of information collected from various other users having similar tastes or preferences.
 - If X and Y have a certain reaction for some items, then they might have the same opinion for other items too. 

#### Advantages & Disadvantages
 - Advantages
   - No domain knowledge necessary
   - Serendipity
   - Great starting point
 - Disadvantages
   - Cannot handle fresh items
   - Hard to include side feature for query/item

## Scoring, and Re-ranking 
 - After candidate generation, another model scores and ranks the generated candidates to select the set of item to display
 - The system combines different sources into a common pool of candidates that are then scored by a single model and ranked according to that score. For example, the system can train a model to predict the probability of a user watching a video on YouTube given the following:
   - query features (for example, user watch history, language, country, time)
   - video features (for example, title, tags, video embedding)
 - The system can then rank the videos in the pool of candidates according to the prediction of the model.
***
**Re-ranking - Freshness**
 - In the final stage of a recommendation system, the system can re-rank the candidates to consider additional criteria or constraints. One re-ranking approach is to use filters that remove some candidates.
 - Another re-ranking approach is to manually transform the score returned by the ranker.
 - Freshness, diversity, and fairness can help improve the recommendation system.
 - Most recommendation systems aim to incorporate the latest usage information. Keeping the model fresh helps the model make good recommendations.
   - Re-run training as often as possible to learn on the latest training data.
   - Create an “average” user to represent new users in matrix factorization models.
   - Use a DNN such as a softmax model
   - Add document age as a feature

**Re-ranking - Diversity**
 - If the system always recommend items that are "closest" to the query embedding, the candidates tend to be very similar to each other. This lack of diversity can cause a bad or boring user experience.
   - Train multiple candidate generators using different sources
   - Train multiple rankers using different objective functions
   - Re-rank items based on other metadata to ensure diversity

**Re-ranking - Fairness**
 - The model should treat all users fairly. 
   - Include diverse perspectives in design and development
   - Train ML models on comprehensive data sets Add auxiliary data when the data is too sparse
   - Track metrics on each demographic to watch for biases
   - Make separate models for underserved groups
