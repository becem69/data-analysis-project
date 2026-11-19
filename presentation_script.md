# Presentation script — Why This Method?

Speaker notes read slide by slide. The same text is embedded in `Why_This_Method.pptx` (Presenter View → Notes).

## Slide 1

Good morning. Today I want to walk you through not just what I did, but why I chose each method. The question I started from was simple: an executive does not want another descriptive report — they want to know why sales move and how to steer them tomorrow. So the title of this talk is "Why This Method?"

Look at the figure behind me on the right. That is the final output of the analysis: an SVM decision boundary and its ROC curve. Everything you see before that slide — the regression diagnostics, the principal components, the correspondence analysis — exists to justify that one decision.

I will show you four things: first, why raw OLS had to be rejected; second, why PCA was needed to understand the structure; third, why MCA was needed for the categories; and fourth, why the SVM is the only method that actually scores a new order. Two thousand eight hundred and twenty-three transactions, four methods, one decision. Let's begin.

## Slide 2

Let me set the scene. The brief was not "describe the data". It was, and I quote: "I no longer want to know what happened. I want to understand why and how to steer sales from tomorrow." That single sentence changes the method entirely.

If the question is only descriptive, a mean and a histogram are enough. But if the question is causal and operational, we need a chain of evidence. Here is the map on the right: we start with an audit, then MCO — the French acronym for OLS — then ACP for PCA, then ACM for MCA, and finally the SVM, which is the only module that produces an actionable probability for a new order.

The code window shows the imports in project.py. This matters because it tells you something important: everything runs in one reproducible local script. There is no notebook, no Colab, no manual step. Every figure you will see today is saved automatically to the images folder, and every number I quote is in output.txt. The method is not just statistically sound — it is repeatable.

## Slide 3

Before any model, we have to trust the data. The file is sales_data_sample.csv, 2,823 rows and 25 columns, read with latin-1 encoding because it contains European characters. In the eight variables we actually use, there are no missing values, so we keep all 2,823 observations.

There is one exception: TERRITORY has 1,074 missing values. I impute those as NA_Americas, but only for the MCA stage — I never let that imputation contaminate the regression.

The code window shows the cleaning step. Notice the last line: I check the correlation between SALES and quantity times price. It is 0.864, not 1. That is a crucial warning. Sales is almost, but not exactly, an arithmetic identity. That already tells us the R-squared we are about to see will be flattering.

The histograms on the right show why we need care: SALES is strongly right-skewed, PRICEEACH is left-skewed because of a discount cap at 100, and neither looks remotely normal. Thirteen observations are multivariate outliers by Mahalanobis distance — only 0.46 percent. I deliberately kept them, because removing them would change the very distribution we are trying to understand. I simply document them.

## Slide 4

This is the slide where the model choice is decided. Before we read a single coefficient, we look at the correlation matrix. Two numbers matter.

First, quantity and price correlate at 0.006 — essentially zero. That is good news: price and volume are independent levers. We can move one without mechanically moving the other, so the business can manage them separately.

Second, PRICEEACH and MSRP correlate at 0.671. That is moderate collinearity, and it produces a VIF of about 1.82. It is below the usual danger threshold of 5 or 10, but it is not zero — and it means the individual price coefficients are less stable than the overall fit suggests.

The heatmap makes this visible instantly: one near-white cell for quantity versus price, one warm cell for price versus MSRP.

I also ran a chi-square test between PRODUCTLINE and DEALSIZE. It is significant — chi-square 135.36, p around 5 times 10 to the minus 23 — but Cramer's V is only 0.155. Statistically detectable, commercially weak. That is a recurring theme today: significance is not the same as importance.

## Slide 5

Here is raw OLS, and here is why I rejected it. On paper it looks excellent: R-squared 0.796, F-statistic around 3,665, all three coefficients highly significant. If I stopped here I would tell you that price and volume both drive revenue. But that conclusion rests on assumptions, and the assumptions fail.

The residuals are not normal: Jarque-Bera is 10,245. They are not homoscedastic: Breusch-Pagan is essentially zero. Durbin-Watson is 1.578, but I deliberately do not interpret it — the rows have no time order, so an autocorrelation test has no meaning here.

Look at the two plots on the right. The residuals-versus-fitted plot fans out: the spread grows with the fitted value. The QQ plot bends away from the diagonal with heavy tails. Kurtosis is 11.48.

And remember the identity problem: SALES is nearly quantity times price. So a high R-squared is partly mechanical. This is why I treat OLS as a diagnostic stage, not as the steering rule.

## Slide 6

So how do we fix it without throwing away the data? Two changes.

First, HC3 standard errors. They do not change the coefficients, but they correct the standard errors for heteroscedasticity, so the t-tests become trustworthy again. That directly addresses the Breusch-Pagan failure.

Second, a log-log specification. Instead of modelling euros, we model elasticities: a one-percent change in quantity or price produces roughly a one-percent change in sales. R-squared rises to 0.906, and the coefficients become scale-free and easier to compare across variables.

The elasticities are about 1.00 for quantity, 1.06 for price and 0.28 for MSRP. In plain language: volume and unit price both matter approximately proportionally, while the catalog price adds a modest independent effect — which is really the discount story.

But please note the caveat. Because price and quantity do not co-move, these are composition elasticities, not demand elasticities. They tell us how revenue is assembled, not what happens if we change price and customers react. And residual non-normality persists even here, so prediction intervals should be treated with caution. OLS survives — but as a diagnostic, not as the final rule.

## Slide 7

Now we move from explaining coefficients to understanding structure. The first rule of PCA is non-negotiable: standardise. Here the scales are wildly different — SALES has a standard deviation around 1,842, quantity around 9.7. Without standardisation, the variance would simply measure the units, and the first axis would be a restatement of SALES.

After standardisation we work on the correlation matrix, whose total inertia is exactly 3. The eigenvalues are 1.6716, 1.0001 and 0.3294. Two axes are above the Kaiser threshold of 1, and together they retain 89.02 percent of the inertia — visible on the scree plot on the right.

That is a strong result. It means three variables reduce to two clean dimensions without losing much. And crucially, we fit this PCA on the three explanatory variables only. SALES never enters the regressors. If we included it, we would be leaking the answer into the predictors and the principal component regression would be circular. That is a design decision, not an accident.

## Slide 8

Why try a principal component regression at all? Because PCA rotates the correlated variables into orthogonal axes. If collinearity were the problem, orthogonal components should fix it — and they do. The VIF of each component is exactly 1.00, by construction.

But look at the result: R-squared is 0.794, slightly worse than raw OLS at 0.796. There is no gain. The reason is interpretability. PC1 is a unit-value axis — price and MSRP together. PC2 is a volume axis. But the third axis, the one we drop, is the contrast between price and MSRP — in other words, discount depth. That is exactly the commercially interesting information, and PCR throws it away.

So the honest conclusion is: VIF of 1.0 here proves only that the components are orthogonal. It does not prove a better model. PCR is a useful check, not a replacement. We keep the log-log OLS as our explanatory specification.

On the right you see the correlation circle with the 2,823 transactions. That is where the real insight is.

## Slide 9

This slide is the payoff of PCA. On the left, the correlation circle; on the right, the individuals plane; and behind them, the biplot. Together these explain 88.66 percent of the inertia.

Read the geometry. The horizontal axis is price and value — PRICEEACH and MSRP point strongly right. The vertical axis is volume — QUANTITYORDERED points up. Sales sits between them, because it is built from both.

The key geometric fact: price and volume arrows are almost at ninety degrees. Ninety degrees means zero correlation. That is the visual confirmation of the 0.006 we saw earlier — and it is why I can tell the business these are independent levers.

Each dot is a transaction. A dot in the upper-right is a high-volume, high-price order — the valuable few. A dot near the origin is a small, cheap order. The biplot lets us read variables and individuals at the same time, which is Benzécri's principle: an individual is interpreted by the arrows it points toward.

This is useful, but it only handles numbers. Our dataset also has product lines, deal sizes, countries and quarters. PCA cannot read those — which brings us to the next method.

## Slide 10

Why can't we just code categories as 1, 2, 3 and run PCA? Because that invents an order and a distance that do not exist. CLASSIC CARS is not "less than" MOTORCYCLES. Treat those codes as numbers and every result is an artefact.

The correct solution is a disjunctive table: each modality becomes its own binary column. With four variables and 18 modalities we get a 2,823 by 18 matrix of zeros and ones, and we analyse it with the chi-square metric. That metric is important — it automatically gives more weight to rare modalities, so a small but distinctive segment is not drowned out by the majority.

The formula on the right is Benzécri's correction. Why do we need it? Because a disjunctive table is sparse: its average eigenvalue is about 1 divided by the number of variables, here 0.25. Raw inertia therefore looks almost flat — 8.65 percent, 8.37 percent, and so on. If we stopped there we would conclude there is no structure. That conclusion would be wrong, and the correction is what reveals the structure.

## Slide 11

This is the slide where the correction changes the story. After Benzécri's adjustment, the first two axes carry 45.22 plus 30.14, or 75.36 percent of the useful inertia. The flat scree plot becomes a clear two-axis structure.

Axis 1 is a commercial-value axis. Its drivers are CLASSIC CARS with a 24.02 percent contribution, LARGE deals at 21.40 percent and SMALL deals at 18.41 percent. Notice something subtle: I compute contributions using the chi-square weight times the squared coordinate, divided by the eigenvalue. If I looked at frequency alone, the MODERATE deal size would win simply because it is common. Contribution corrects for that — it measures what actually builds the axis.

Axis 2 is spatio-temporal: APAC, EMEA, NA_Americas and quarter T3. So the qualitative space has two clean readings — what is sold and how big the deal is, then where and when.

That is a segmentation we can act on. But eighteen modalities on a plane is still hard to operationalise, so I need one more step to turn it into groups.

## Slide 12

This slide explains why MCA and hierarchical clustering belong together. MCA gives us denoised coordinates in a low-dimensional space; Ward's algorithm turns those coordinates into actual segments. Clustering directly on the raw binary columns would be dominated by noise and by frequent modalities. Clustering on the first five MCA axes uses the structure we just validated.

The dendrogram on the left groups the 18 modalities into four actionable segments. The most important one is the combination of CLASSIC CARS and LARGE deals — a clear key-account profile. The other groups are more fragmented, volume-driven and seasonal.

The code window shows just how little is required: one call to linkage with Ward's method, one dendrogram. This is worth emphasising: the analytical work was in choosing the space, not in running the algorithm.

Now we have four things: a rejected raw regression, a valid log-log explanation, an orthogonal quantitative structure, and a qualitative segmentation. What we do not yet have is a way to score a new order. That is the final method.

## Slide 13

Here is the chosen final method: a support vector machine with an RBF kernel. Why a support vector machine rather than another regression? Because the question has changed. We no longer ask how much sales will be — we ask whether an order will be LARGE, and we want a probability we can plug into a CRM.

Three implementation choices matter. First, the target is IS_LARGE, derived from DEALSIZE. The minority class is only 5.56 percent, so I use a stratified 75-25 split to keep the rare cases represented in the test set. Second, the scaler is fitted on the training fold only — no leakage from the test set. Third, C equals 10 and gamma is scale, which keeps the margin reasonably wide while letting the boundary bend.

The results are excellent: 99.58 percent accuracy, AUC 0.9999, and an F1 of 0.961 on the rare LARGE class. I show the RBF decision boundary because it is the visual proof that a non-linear boundary is being used.

But a careful analyst does not stop at an excellent number. Let me show you why.

## Slide 14

Why did I not simply trust the 99 percent? Because of a circularity problem, and because accuracy is a trap with an imbalanced target.

Look at the warning in the code window. DEALSIZE is not an independent outcome — it is a deterministic binning of SALES: small below 3,000, medium between 3,000 and 7,000, large above 7,000. So if I give the model SALES as a feature, it partly relearns a business rule rather than discovering a pattern. I am explicit about that in the script.

The comparison table makes the point. A dummy classifier that always predicts NON-LARGE scores 94.48 percent accuracy — with zero detections. Our RBF model scores 99.58 percent with an F1 of 0.961, so it is genuinely useful, but accuracy alone would have hidden the difference.

A linear SVM actually reaches 100 percent here, so the RBF kernel's superiority is not demonstrated — I keep it as a prudent default. And when I remove SALES entirely, accuracy drops to 97.31 percent and F1 to 0.716. That is the honest estimate of what the model can do with genuinely pre-order information.

So the recommendation is conditional: deploy, but validate against the linear baseline and monitor for drift.

## Slide 15

Let me dwell on this figure for a moment, because it is the evidence I would leave on the table if you remember only one slide.

On the right is the ROC curve on the 706 test orders. An AUC of 0.9999 means that a randomly chosen LARGE order is almost always ranked above a randomly chosen NON-LARGE order. That is what makes the model useful for lead scoring: we can rank a queue, not just label it.

On the left is the decision boundary, projected onto two principal components so we can draw it. The warm region is LARGE, the cool region is NON-LARGE, and the dashed line is the frontier. It is curved — that is the RBF kernel doing its work.

The confusion matrix is 666 true negatives, one false positive, two false negatives and 37 true positives, out of only 39 LARGE orders. Missing two of 39 is acceptable in this context; missing zero would be suspicious. This is the visual, quantitative proof that the chosen method works.

## Slide 16

So what do we do with it? Four operational rules.

A, lead scoring: the model returns a probability in real time inside the CRM, so we can flag orders that are 'large in the making' before they close.

B, key accounts: when an order is confidently LARGE, attach dedicated service and priority logistics. That is the Pareto logic — focus on the valuable few.

C, upsell: target profiles sitting near the threshold with volume tiers and discounts, because we know from the PCA that volume and price are independent levers we can pull separately.

D, automation: route confidently NON-LARGE orders to straight-through invoicing, so human attention goes where it earns its keep.

The statistics behind this: 5.56 percent of orders are LARGE, and 97 of the 157 Large orders are CLASSIC CARS. That concentration is exactly why this is worth automating. Note that these actions follow from the whole chain — the segmentation told us who matters, the elasticities told us which levers exist, and the SVM tells us which individual order to act on.

## Slide 17

Let me summarise the chain of evidence, because that is really the answer to 'why this method'.

Can we trust price effects? We tested with MCO plus HC3, found a VIF of 1.82 and Jarque-Bera of 10,245 — and rejected the raw OLS. That rejection is what forced the log-log model.

What is the real structure? ACP and PCR gave us 88.66 percent of inertia and a VIF of 1.0 — telling us price and volume are separate levers, and that PCR adds nothing.

Which segments exist? ACM with Benzécri correction gave 75.36 percent of useful inertia and four classes, which created the targeting rules.

Can we score a new order? The RBF SVM gave AUC 0.9999 and F1 0.961 — and that is why it is the chosen final method.

The key message: the SVM was not chosen in isolation. It rests on the rejected OLS, the orthogonal space of the PCA, and the segments of the MCA. And I present it with its limits — circular target, strong linear benchmark, and the need for drift monitoring. A method is only as good as the critique it survives.

## Slide 18

That concludes the presentation. The full analysis is in project.py, the complete English console output is in output.txt, all nine figures are in the images folder, and the answers to the sixteen critical questions from the workbook are in reponses_questions.md.

If you take one thing away, let it be this: the best method is rarely the most sophisticated one. It is the one whose assumptions you have tested, whose limitations you can state out loud, and whose output someone can actually use tomorrow.

Thank you. I am Bacem El Manai, github.com/becem69. I would be happy to take your questions.

