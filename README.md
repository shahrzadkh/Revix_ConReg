# Revix_ConReg
This is a repository for codes used in the implementation of CogReg, a topic modelling App for legal acts. 
## Data Analysis:
### Text Preprocessing:
- Performed using **NLTK** and **sklearn** within python. 
### Topic modelling:
- performed using **Genism**, but functions for creating Feature vectors and performing subsequent topic modelling in **Sklearn** is also provided in the notebook. 
### Visualization: 
- _Word cloud_
- _Topic distance map_ using **pyLDAvis**
- Network of **_Topic to Article_** Connections using **NetwrkX**, **Bokeh**, and **Holoviews**
** **
_See "Copy_of_Topic_modelling_with_Genism_for_Binder_final.ipynb" for final version of the used code and also all additional functions for generating the topic modelling in sklearn and Genism using both LDA and LSI._
** **

## User Interface: 
- Generated with **Anvil**

** **
_Source code of the Anvil app: ConRegReViX.yaml_
** **
# ToDos:
- Clean up the links between the data-table and the Pdf, so as by clicking on the links, the user woll see the relevant Article's text only (with no html code). 
- Clarify why Chrome does not allow reloading of the iframe, thus does not allow the users to see the second intercative graph. 
- Use more beautiful Graph visualisations, such as neo4j.
