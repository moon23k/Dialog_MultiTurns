## Multi Turn Dialog Generation Study

Dialogue Generation Task can be divided into single turn and multi turn dialogue generation based on the number of dialogues. 
Multi-turn is a more difficult task than single-turn because it has to generate an answer while considering the context of the previous conversation.
This repo will explore mainly three ways of generate multi-turn dialogue responses.
each is expanded single turn, hierarchical, consider all at once.

<br><br>

## Strategies
**expanded single turn** <br>
The biggest difference between a multi turn and a single turn is whether or not it reflects the previous conversation.
In other words, the multi turn dialogue generation task can be viewed as an extended single turn dialogue generation task.
While learning the maximum number of turns, this method aims to gradually develop multi-turn abilities.

<br> **Hierarchical** <br>


<br> **Cover Long Dialogues History at Once** <br>


<br><br>

## Experimental Setup

**Model Setup** <br>

**Data Setup** <br>

**Training Setup** <br>

<br><br>

## Results


<br><br>

## How to Use
```
git clone 
```
```
python3 setup.py
```
```
python3 run.py -mode [train, test, inference] -strategy [extended, hierarchical, long]
```

<br><br>

## References
