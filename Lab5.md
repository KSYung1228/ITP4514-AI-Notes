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
 - Example

