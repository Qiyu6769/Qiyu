# THIS FILE IS IN THE HANDOUTS FOLDER. COPY IT INTO YOUR CLASS NOTES

- [**Read the chapter on the website!**](https://ledatascifi.github.io/ledatascifi-2023/content/05/02_reg.html) It contains a lot of extra information we won't cover in class extensively.
- After reading that, I recommend [this webpage as a complimentary place to get additional intuition.](https://aeturrell.github.io/coding-for-economists/econmt-regression.html)

## Today

[Finish picking teams and declare initial project interests in the project sheet](https://docs.google.com/spreadsheets/d/1kRbuRKfKh9lCdoVBGLxSbDTIRBEfnV7Y8AcP-hZbmTw/edit#gid=1508330834)


# Today is mostly about INTERPRETING COEFFICIENTS (6.4 in the book)

1. 25 min reading groups: Talk/read through two regression pages (6.3 and 6.4) 
    - Assemble your own notes. Perhaps in the "Module 4 notes" file, but you can do this in any file you want.
    - After class, each group will email their notes to the TA/me for participation. (Effort grading.)
1. 10 min: class builds joint "big takeaways and nuanced observations" 
1. 5 min: Interpret models 1-2 as class as practice. 
1. 20 min reading groups: Work through remaining problems below.
1. 10 min: wrap up  

---

- Goodness of fit 6.3
 - R2 is SSE/TSS. 0 to 1 usually
 - adj R2 penalizing you for adding more variables
 - what's a "good R2" ... depends on the y variable. check other studies for a baseline.
- regression is: $y = a+b_1*x_1+ ...$
- the relationship between y and x_1 is $\delta y/ \delta x_1 = \beta_1$
- interpretation:
 - 
 - depends on X: continuous, binary, categorical
 - if continuous:


```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col # nicer tables

```


```python
url = 'https://github.com/LeDataSciFi/ledatascifi-2023/blob/main/data/Fannie_Mae_Plus_Data.gzip?raw=true'
fannie_mae = pd.read_csv(url,compression='gzip') 
```

## Clean the data and create variables you want


```python
fannie_mae = (fannie_mae
                  # create variables
                  .assign(l_credscore = np.log(fannie_mae['Borrower_Credit_Score_at_Origination']),
                          l_LTV = np.log(fannie_mae['Original_LTV_(OLTV)']),
                          l_int = np.log(fannie_mae['Original_Interest_Rate']),
                          Origination_Date = lambda x: pd.to_datetime(x['Origination_Date']),
                          Origination_Year = lambda x: x['Origination_Date'].dt.year,
                          const = 1
                         )
                  .rename(columns={'Original_Interest_Rate':'int'}) # shorter name will help the table formatting
             )

# create a categorical credit bin var with "pd.cut()"
fannie_mae['creditbins']= pd.cut(fannie_mae['Co-borrower_credit_score_at_origination'],
                                 [0,579,669,739,799,850],
                                 labels=['Very Poor','Fair','Good','Very Good','Exceptional'])

```

### Statsmodels

As before, the psuedocode:
```python
model = sm_ols(<formula>, data=<dataframe>)
result=model.fit()

# you use result to print summary, get predicted values (.predict) or residuals (.resid)
```

Now, let's save each regression's result with a different name, and below this, output them all in one nice table:


```python
# one var: 'y ~ x' means fit y = a + b*X

reg1 = sm_ols('int ~  Borrower_Credit_Score_at_Origination ', data=fannie_mae).fit()

reg1b= sm_ols('int ~  l_credscore  ',  data=fannie_mae).fit()

reg1c= sm_ols('l_int ~  Borrower_Credit_Score_at_Origination  ',  data=fannie_mae).fit()

reg1d= sm_ols('l_int ~  l_credscore  ',  data=fannie_mae).fit()

# multiple variables: just add them to the formula
# 'y ~ x1 + x2' means fit y = a + b*x1 + c*x2
reg2 = sm_ols('int ~  l_credscore + l_LTV ',  data=fannie_mae).fit()

# interaction terms: Just use *
# Note: always include each variable separately too! (not just x1*x2, but x1+x2+x1*x2)
reg3 = sm_ols('int ~  l_credscore + l_LTV + l_credscore*l_LTV',  data=fannie_mae).fit()
      
# categorical dummies: C() 
reg4 = sm_ols('int ~  C(creditbins)  ',  data=fannie_mae).fit()

reg5 = sm_ols('int ~  C(creditbins)  -1', data=fannie_mae).fit()

```

Ok, time to output them:


```python
# now I'll format an output table
# I'd like to include extra info in the table (not just coefficients)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

# q4b1 and q4b2 name the dummies differently in the table, so this is a silly fix
reg4.model.exog_names[1:] = reg5.model.exog_names[1:]

# This summary col function combines a bunch of regressions into one nice table
print('='*108)
print('                  y = interest rate if not specified, log(interest rate else)')
print(summary_col(results=[reg1,reg1b,reg1c,reg1d,reg2,reg3,reg4,reg5], # list the result obj here
                  float_format='%0.2f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3 (log)','4 (log)','5','6','7','8'], # these are bad names, lol. Usually, just use the y variable name
                  info_dict=info_dict,
                  regressor_order=[ 'Intercept','Borrower_Credit_Score_at_Origination','l_credscore','l_LTV','l_credscore:l_LTV',
                                  'C(creditbins)[Very Poor]','C(creditbins)[Fair]','C(creditbins)[Good]','C(creditbins)[Vrey Good]','C(creditbins)[Exceptional]']
                  )
     )
```

    ============================================================================================================
                      y = interest rate if not specified, log(interest rate else)
    
    ============================================================================================================
                                            1        2      3 (log) 4 (log)     5         6        7        8   
    ------------------------------------------------------------------------------------------------------------
    Intercept                            11.58*** 45.37*** 2.87***  9.50***  44.13*** -16.81*** 6.65***         
                                         (0.05)   (0.29)   (0.01)   (0.06)   (0.30)   (4.11)    (0.08)          
    Borrower_Credit_Score_at_Origination -0.01***          -0.00***                                             
                                         (0.00)            (0.00)                                               
    l_credscore                                   -6.07***          -1.19*** -5.99*** 3.22***                   
                                                  (0.04)            (0.01)   (0.04)   (0.62)                    
    l_LTV                                                                    0.15***  14.61***                  
                                                                             (0.01)   (0.97)                    
    l_credscore:l_LTV                                                                 -2.18***                  
                                                                                      (0.15)                    
    C(creditbins)[Very Poor]                                                                             6.65***
                                                                                                         (0.08) 
    C(creditbins)[Fair]                                                                         -0.63*** 6.02***
                                                                                                (0.08)   (0.02) 
    C(creditbins)[Good]                                                                         -1.17*** 5.48***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Exceptional]                                                                  -2.25*** 4.40***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Very Good]                                                                    -1.65*** 5.00***
                                                                                                (0.08)   (0.01) 
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared Adj.                       0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    Adj R-squared                        0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    No. observations                     134481   134481   134481   134481   134481   134481    67366    67366  
    ============================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    

### NOTES:

$y = -16.81 + 3.22lgc + 14.61lgl -2.18lgc*lgl$

in column 7, if we have several different credits, this is a binary variable.

in column 8, it is an average group category

easy to show regression, standard error in(), $t-stat = coef/se$

the lower the p is, the more "certain" we can be that the relationship isn't 0

in regression model, 1 star means p<10%, 2 stars means p<5%, 3 stars means p<1%


### a statistically significant relationship between x AND Y

good trick: scale continuous variables by their STD so that a one unit change in X is a STD (TEXT BOOK: 6.4.7)

purpose of this table: not predicting but the relationship between X and Y.

# Today. Work in groups. Refer to the lectures. 

You might need to print out a few individual regressions with more decimals.

1. Interpret coefs in model 1-4
1. Interpret coefs in model 5
1. Interpret coefs in model 6 (and visually?)
1. Interpret coefs in model 7 (and visually? + comp to table)
1. Interpret coefs in model 8 (and visually? + comp to table)
1. Add l_LTV  to Model 8 and interpret (and visually?)






```python
fannie_mae['int'].describe()
```




    count    135038.000000
    mean          5.238376
    std           1.289895
    min           2.250000
    25%           4.250000
    50%           5.250000
    75%           6.125000
    max          11.000000
    Name: int, dtype: float64




```python
-0.4276/fannie_mae['int'].std()
```




    -0.3314997324254188




```python
reg1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>int</td>       <th>  R-squared:         </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.938e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 29 Mar 2023</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>10:53:10</td>     <th>  Log-Likelihood:    </th> <td>-2.1575e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>134481</td>      <th>  AIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>134479</td>      <th>  BIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                            <td>   11.5819</td> <td>    0.046</td> <td>  253.270</td> <td> 0.000</td> <td>   11.492</td> <td>   11.671</td>
</tr>
<tr>
  <th>Borrower_Credit_Score_at_Origination</th> <td>   -0.0086</td> <td> 6.14e-05</td> <td> -139.198</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.008</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2660.479</td> <th>  Durbin-Watson:     </th> <td>   0.397</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2660.737</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.321</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.750</td>  <th>  Cond. No.          </th> <td>1.04e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.04e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### model explanation

- model 1:
    - when credit score goes up by 1, the interest rate on the loan decreases by 0.86 (bp)
    - if my credit score from 700 to 707, interest rate falls by 0.0602 (6 bp)
- model 2:
    - when credit score goes up by 1, the interest rate on the loan decreases by 0.0607 (bp) 
    - if my credit score from 700 to 707, interest rate falls by 0.0607 (6.07 bp)
- model 3:
    - when credit score goes up by 1, the interest rate on the loan decreases by 0.17% (bp) 
    - if my credit score from 700 to 707, interest rate falls by 0.0607 (6.07 bp)
- model 4:
    - when credit score goes up by 1, the interest rate on the loan decreases by 1.19% (bp) 
    - if my credit score from 700 to 707, interest rate falls by 0.066 (6.6 bp)  $5.59 - 5.59*(1-0.0119)$ # lose 1 
- model 5:
    - when credit score goes up by 1, the interest rate on the loan decreases by 0.0599 (bp) 
    - if my credit score from 700 to 707, interest rate falls by 0.0607 (6.07 bp)


```python
5.59 - 5.59*(1-0.0119) # lose 1 
```




    0.06652099999999983




```python
reg1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>int</td>       <th>  R-squared:         </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.938e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 29 Mar 2023</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>10:53:18</td>     <th>  Log-Likelihood:    </th> <td>-2.1575e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>134481</td>      <th>  AIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>134479</td>      <th>  BIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                            <td>   11.5819</td> <td>    0.046</td> <td>  253.270</td> <td> 0.000</td> <td>   11.492</td> <td>   11.671</td>
</tr>
<tr>
  <th>Borrower_Credit_Score_at_Origination</th> <td>   -0.0086</td> <td> 6.14e-05</td> <td> -139.198</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.008</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2660.479</td> <th>  Durbin-Watson:     </th> <td>   0.397</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2660.737</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.321</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.750</td>  <th>  Cond. No.          </th> <td>1.04e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.04e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
reg2.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>int</td>       <th>  R-squared:         </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   9656.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 27 Mar 2023</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:00:32</td>     <th>  Log-Likelihood:    </th> <td>-2.1578e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>134481</td>      <th>  AIC:               </th>  <td>4.316e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>134478</td>      <th>  BIC:               </th>  <td>4.316e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>   44.1324</td> <td>    0.302</td> <td>  145.949</td> <td> 0.000</td> <td>   43.540</td> <td>   44.725</td>
</tr>
<tr>
  <th>l_credscore</th> <td>   -5.9859</td> <td>    0.044</td> <td> -134.888</td> <td> 0.000</td> <td>   -6.073</td> <td>   -5.899</td>
</tr>
<tr>
  <th>l_LTV</th>       <td>    0.1546</td> <td>    0.010</td> <td>   14.765</td> <td> 0.000</td> <td>    0.134</td> <td>    0.175</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2793.369</td> <th>  Durbin-Watson:     </th> <td>   0.386</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2743.990</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.321</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.720</td>  <th>  Cond. No.          </th> <td>    735.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python

```


```python

```
