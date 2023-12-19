# Lab5 - Uncertainty and Probailistic Reasoning

## Uncertainty
 - In AI, when an agent knows enough facts about its environment, the logical plant and cations produces a guaranteed work
 - Unfortunately, agents never have access to the whole truth about their environment. Agents act under uncertainty
 - Example for uncertainty:
   - Weather Forecast
   - Financial Market
   - Health Diagnosis
   - Sport Outcome
 - Major causes of uncertainty to occur in the real world
   - Information occurred from unreliable sources;
   - Experimental errors;
   - Equipment fault;
   - Temperature variation;
   - Climate change

## Dealing with Uncertainties
 - Probabilistic Reasoning: Agents can use probabilistic models, sch as Bayesian networks or Markov decision processes, to reason under uncertainty
 - Monte Carlo Methods: such as Monte Carlo simulation or monte Carlo tree search
 - Enemble Learning: combine multiple models or predictions such as bagging or boosting

## Probability vs Artifical Intelligence
 - Data is important to gather AI ability
 - In analytics process, we usually use random variables to describe the data
   - Mathematical process to slove AI problems
   - To make the data(input) computable
   - To find the patterns behind data(for decison making)
   - Allows the ana,ysis of satistics(partical)
 - Probability helps predict likely that an event will hapen
   - E.g., weather prediction, product recommendation

## Probabilistic Reasoning
 - ***Probabilistic reasoning*** is a way of knowledge represemtation where we apply the concept of ***probability*** to indicate the ***uncertainty in knowledge***
 - In probailistic reasoning, we combine probability theory with logic to handle the uncertainty
 - Probability provides a way to habdle the uncertainty that is the result of someone's laziness and ignorance
 - In real world, lots of senarios, where the certainty of somethings is not confirmed:
   - "It will rain today"
   - "Behavior of someone for some situations"
   - "A match between two teams or two players"
 - These are probable sentences for which we can assume that it will happen but not sure about it, si here we use probabilistic reasoning

## Sloving Problem with Uncertainty
 - Bayes' rule
 - Bayesian Satistics

## Probability
 - Numerical description of how likely an event is to occur and or how likely that a proposition is true
 - When run a random experiment N times, during which an event A occurs m times, then we say the frequencr of A's occurrence is $ f_A = \frac{m}{n}$
 - When n is large enough, $f_A$ will be very close to a value p, which is defined as the probability of A to occue, i.e., $\lim\limits_{n \to \infty} f_A \equiv P(A) = p$
   - Law of large numbers
   - When we toss a coin, the probability of "heads up" is 0.5

## Sample space
 - The set of all possible outcomes of an experiment
 - Example: coin flapping
   - Possible outcomes: 
     - Hands up - H
     - Hands Down - T
   - The sample space: S = {H,T}
   - For fair coins
     - What are the chances of H and T happening
     - H and T have 50-50 chance to happen
     - The probability for H to happen is 50%(0.5), so does T

## Rolling dice
 - Sapce S = {1, 2, 3, 4, 5 ,6}
 - Example event E = {1, 3, 4}
 - If we roll the dice once, and it lands with 1/3/4, then we say the event E occurs
 - The probability that occurs is : $P(E) = \frac{1+1+1}{6} = \frac{3}{6} = \frac{1}{2}$

 - We can do set operations fot events
 - If E and F are events, then we can form:
   - $E^c$ --- the complement of E
   - $E\cup F$ --- The union of E and F
   - $E\cap F$ --- The intersection of E and F
   - $E \sqsubseteq F$ --- E is the subset of F
 if the smaple space S is dinite(able to list the elements), we alow all subsets to be an event 

## Basic Probaility
 - *Probability* can be defined as a chance that an ***uncertain event*** will occur
 - It is the numerical measre of the likelihood that an event will occur
 - The value of probability always remains between 0 and 1that pepresent idea uncertainties
 - The probability of an uncertain event:
$$
Probability\  of ocurrence \ = \frac{Number of desired outcomes}{Total number of outcomes}
$$
 - Some basic properties of probability:
   - $0 \le P(A) \le 1$, where P(A) is the probability of an event A
   - $P(A) = 0$, indicates total uncertainty in an event A
   - $P(A) = 1$, indicates total certainty in an event A
   - $P(﹁A) = $probability of a not happening event
   - $P(﹁A)+P(A) = 1$
 - Basic terms in probability:
   - Event
   - Sample space
   - Random variables
   - Prior probability
 - Posterior Probability: calcualted after all evidence or information has taken into account. it is a combination of proir parobabilityand new information $P(Rain|Cloud) = \frac{P(Cloud|Rain)P(Rain)}{P(cloud)}$ 

## Conditional Probability
 - Conditional probability is a probaility of occurring an event when another event has already happened
 - Suppose we want to calculate the event A when event B has already occurred, "the probability of A under the conditions of B", it can be written as:$P(A|B) = \frac {P(A \land B)}{P(B)}$, where $P(A \land B) = $ Joint probability of A and B;P(B) = Marginal probility of B
 - If probability A is given and need to find B, it will be given as: $P(B|A) = \frac{P(A \land B)}{P(A)}$
 - It can explained by using the left Venn diagram, where B is occurred event, so sample space will be reduced to set B
 - It can be only calculate event A when event B is already occrred by diving the probability of P(A∧B) by P(B)

Example: 
 - 70%Student like JAVA, 40%Student likes and 40% student likes java and Python, and then what atA  A
 - Solution:
   - Let A be an event that a student likes Python; and B be an event that a stdent likes Java 
   $$
   P(A|B) = \frac{P(A \land B)}{P(B)} = \frac{0.4}{0.7} = 57\%
   $$
   - So, 57% are the students who like Java also like Python

## Bayes' Theorem
 - Bayes' theorem is also known as Bayes' rule, Bayes' law, or Bayesian reasoning, which determines the probability of an event with uncertian knowledge
 - In probability theory, it relates the conditional probabbility and marginal probabilities of two random events
 - Bayes' theorem was named after the british mathematician Thomas Bayes
 - The Bayesian inference is an application of Bayes; theorem, which is funfamental to Bayesian statistics
 - Bayes' Theorem gives us a way to valvulate the value of P(A|B) which the knowledge of P(B|A)
 - It allows updating the probability prediction of an event by observing new information of the real world\
 - Example:
   - If cancer corresponds to one's age then by using Bayes' theorem, we can determine the probability of cancer more accurately with the help of age
 - Bayes' Theprem can be derived using product rule and conditional probability of event A with known event B:
   - As from product rule we can wirte:
     - $P(A \land B) = P(A|B)P(A)$ or 
   - Similarly, the probabiity of event B with known event A:
     - $P(A \land B) = P(B|A)P(A)$
   - Equating right hand side of both the equation, we will get:
     - $P(A|B) = \frac{P(B|A)*P(A)}{P(B)}$
 - The above equation(a) is called as Bayes'eule or Bayes' theorem
 - This queation is basic of most modern AI system for probabilitic inference
 - Bayes' Theorem shows the simple relationship between joint and conditional probabilities
 - Here, P(A|B) is kown as posterior, which we need to calculate, and it will be read as Probaility of hypothenis A when we have occurred an evidence B
 - P(B|A) is caled the likelihood, in which we consider that hypothesis is true, then we calculate the probability of evidence
 - P(A) is called the prior probability, probability of hypoyhesis before considering the evidence
 - P(B) is called marginal probability, pure probability of an evidence
 - In Equation(1), ingeneral, we can wirte P(B) = P(A) x P(B|Ai), hence the Bayes' rule can be written as:
  $$
  P(A_i|B) = \frac{P(A_i)*P(B|A_i)}{\sum_{i=1}^nP(A_i)*P(B|A_i)}
  $$
  where $A_1,A_2,A_3...,A_n$ is a set of ntaally exclusive and exhaustice events
  ![](/Lab5/Picture1.png)

## Applying Bayes' Rule
 - Bayes' Rule can be written in terms of cause and effect:
$$
P(cause|effect) = \frac{P(effect|cause)*P(cause)}{P(effect)}
$$
 - For example, frm a standard deck of playing cards, a single card is drawn
 - the probability that the card is king is 4/52, then calculate posterior probability $P(king|face)$, which means thedrawn face card s a king card
   - Solution:
     - $P(king|face) = \frac{P(face|king)*P(king)}{P(face)}$
     - P(king): probability that the card is king = 4/52 = 1/13
     - P(face): probability that a card is a face card = 3/13
     - P(face|king): probability of face card when we assume it is a king = 1
     - Putting all values in Equation, we will get :
       - $P(king|face) = \frac{1*\frac{1}{13}}{\frac{3}{13}} = \frac{1}{3}$
     - It is a probability that a face card is a king card
## Use Cases of Bayes' Rule
 - Bayes' theorem allows scientists to combine a priori beliefs about the probability of an event(or an environmental condition, or another metric) with emprical(that is, observation-based) evidence, resulting in a new and more robst posterioe probability distribtion
 - there are plenty of applications of the Bayes' Rule in the real world
 1. Valuating depression tests performance;
 2. Predicting water quality conditions;
 3. Assisting in COVID tests;
 4. Weather forecasting;
 5. Calculating the oving steps od robots;
 6. Solving the Monty Hall problem

## Probability Programming in Pyro
