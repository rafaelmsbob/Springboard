# Overview
Here follows a quick description of the files inside this folder:
* Complete Code.py
<p>This code reads the raw .csv files ("train.csv", "test.csv") from Kaggle,performs Data Wrangling techniques and applies
our best regression model, GradientBoostRegressor().</p>

* Final Presentation.pptx
<p>Powerpoint slides of the main findings and takeaways of the project, derived from the FinalReport.</p>

*  FinalReport (.docx and .pdf formats)
<p>Complete report, with all parts of the project, written in an easy to follow format. You will find the motivation for the project,
visual EDA along with its codes, Data wrangling techniques applied to the raw dataset along with its codes, a full table of all
the Regression techniques applied to the dataset with its scores (log of RMSE and R squared) and a full description of the takeaways.
In addition, there are screenshots of the Regression models tried, as well as a description of all the columns found in the
raw .csv files.</p>

* GoogleDocsLink
<p>.txt file with the link to the FinalReport in a GoogleDocs format.</p>

* GradientBoost-Plot.py
<p>Full code to generate Prediction X Actual values plot. It uses our best model, GradientBoostRegressor(), however the following
modification was needed: as we do not have access to the Actual values of the test.csv dataset (Kaggle does not make this data public),
we reserved 20% of the training dataset to apply our model and compare our predictions with the Actual values. </p>
