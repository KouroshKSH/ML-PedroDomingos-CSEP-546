<h1>Rule Induction</h1>

Why not just directly learn rules, instead of first making a decision tree and then turning it into a set of rules? The answer to this question will come in 2 parts:

1.  Propositional rules
2.  First-order rules



<h1>Learning Sets Of Rules</h1> 

A famous example is from [Walmart](https://www.walmart.com/), where the data miners realized that: 

>   A set of customers who buy diapers, purchase beers as well. :baby: :beer:

The explanation for this situation is that, fathers who are buying diapers for their kids, may tend to buy beers to compensate for the pain of life.

Rules are very easy to understand popular in data mining:

-   **Variable size:** Any boolean function can be presented.
-   **Deterministic:** Rules are just a simple implication.
-   **Discrete and continuous parameters:** Confidence, coverage and numbers like these can be attached to rules.

Learning algorithms for rule sets can be described as:

-   **Constructive search:** The rule set is built by adding rules; each rule is constructed by adding conditions.
-   **Eager:** First give a data base, then extract all the rules possible, and finally apply the rules when the time comes.
-   **Batch:** Rules are mostly learned in batch mode.

The funny thing is that, in the previous example, Walmart decided to put the beer section as far away as possible from the diapers section, in order to maximize the chances of persuading fathers to purchase extra stuff whilst walking the aisle. The actions you take based on the rules are a whole different issue, but the first step is to discover the associations.



<h1>Rule Set Hypothesis Space</h1>

Like decision trees, rules are not a mathematically complicated representation. 

-   **Each rule is a conjunction of tests.** Each test has the form $\ x_j = v,\ x_j \le v $ or $\ x_j \ge v $ where $\ v $ is a value for $\ x_j $ that appears in the training data.
    $$
    x_1 = Sunny \ \and \ x_2 \le 75 \% \Rightarrow y = 1
    $$
    

-   **A rule set is a disjunction of rules.** Typically all of the rules are for one class _(e.g. $\ y = 1 $)_. An example is classified into $\ y = 1 $ if **any** rule is satisfied.
    $$
    \begin{align}
    	x_1 = Sunny \ \and \ x_2 \le 75 \% \Rightarrow y = 1 \\
    	x_1 = Overcast \Rightarrow y = 1 \\
    	x_1 = Rain \ \and \ x_3 \le 20 \Rightarrow y = 1
    \end{align}
    $$



<h1>Relationship To Decision Trees</h1>











