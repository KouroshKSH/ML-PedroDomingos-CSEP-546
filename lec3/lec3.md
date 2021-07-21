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

You can turn a set of rules, into a truth table, and then transform that into a decision tree. Since a decision tree may be larger than a set of rules, you can't just easily turn decision trees into rule sets and vice versa; there's a snag. 

A small set of rules can correspond to a big decision tree, because of the $\ Replication \ Problem $ .
$$
x_1 \and x_2 \Rightarrow y = 1 \qquad \qquad x_3 \and x_4 \Rightarrow y = 1 \qquad \qquad x_5 \and x_6 \Rightarrow y = 1
$$
![image_of_relationship_to_decision_trees](https://raw.githubusercontent.com/LiLSchw4nz/ML-PedroDomingos-CSEP-546/master/images/image_of_relationship_decision_trees.png)

If we allow a [decision graph](https://www.bayesserver.com/docs/introduction/decision-graphs), then a decision tree graph won't suffer from a blob of set of rules. However, these graphs host a number of problems that make them inefficient. 

As we can see, even though the rules are simple, the size of the tree grows exponentially with the number of rules. In general, converting a set of rules into a decision tree might cause an exponential blowup. In this regard, rules have a serious advantage compared to decision trees.



<h1>Learning A Single Rule</h1>

How can we propose a set of rules? For example, we have a bank, and they want to decide whether a customer is a good credit risk. Probably, the first thing to consider is a rule that has a highly predictive feature, such as the person's total net worth in this case.

We grow a rule by starting with an empty rule and adding tests one at a time until the rule **covers** only positive examples.

$\ \begin{aligned}&\large \textbf{GrowRule} (S)\\& R = \{ \ \} \\&\textbf{repeat} \\&\qquad\textrm{choose best test}\ x_j \Theta v \ \textrm{to add to}\  R, \textrm{where}\ \Theta \in \{= , \neq , \leq , \ge \} \\ & \qquad S := S - (\textrm{all examples that do not satisfy}\ R \cup \{ x_j \Theta v \}) \\ & \textbf{until}\ S \ \textrm{contains only positive examples} \end{aligned} $



A question might be asked, what if we have added every single feature to a rule, but there is still a mixture of positive and negative examples? Might this situation happen? **Absolutely!** 

Consider this, we have 2 patients, with exactly the same symptoms. Now, one of them might have the flue, and the other doesn't. This can happen all the time. Therefore, there's a similarity between this and the induction on decision trees. You might have one rule that can capture a certain portion of the population, but what if you want to find out the other people, who can be valid candidates in your research? It's like when a single rule covers a small portion of positive examples _(they were counted for)_, but now there are still a bunch of other positive examples that the rule hasn't covered, but we still need to find. The next thing to do, is to find another rule that can cover as many positive examples as possible, which is accurate as much as possible _(meaning, it covers as less negative examples as possible)_.   
