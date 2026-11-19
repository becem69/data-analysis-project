# Réponses aux questions du PDF « Des questions pour stimuler l'esprit critique »

*Toutes les réponses ci-dessous s'appuient sur les résultats **réellement calculés** par
`analyse_complete.py` à partir de `sales_data_sample.csv` (2 823 transactions). Elles sont
rédigées dans la langue des ateliers. Le détail numérique se trouve dans
`resultats_analyse.txt`. Les appels à Gujarati & Porter (2009), Hastie et al. (2009),
Husson et al. (2017) etc. renvoient aux références citées dans les PDF.*

**Deux mises au point préalables (esprit critique) :**
1. Le PDF MCO annonce une corrélation `PRICEEACH`–`MSRP` de **0,875** dans un commentaire,
   mais la matrice de corrélation du PDF annonce **0,671** et le calcul sur le CSV donne
   **0,6706** (VIF ≈ 1,82). C'est la valeur calculée qui est retenue ici.
2. Le PDF ACM annonce « p = 0,0000056 » pour le chi² `PRODUCTLINE × DEALSIZE` : la valeur
   exacte est **5,16 × 10⁻²³** (chi² = 135,36 ; ddl = 12). La conclusion (rejet
   d'indépendance) reste identique.

---

## 1. Module Économétrique : les Moindres Carrés Ordinaires (MCO)

### 1.1 Pourquoi doit-on impérativement inspecter la matrice de corrélation et détecter les corrélations trop élevées (comme celle entre le prix pratiqué et le prix conseillé) avant d'interpréter les coefficients d'une régression linéaire (Gujarati & Porter, 2009) ?

Parce que le MCO estime les coefficients en inversant la matrice `X'X`. Si deux régresseurs
sont presque proportionnels, `X'X` devient quasi-singulière : son déterminant s'approche de
zéro et sa condition number explose (ici **789** dans le modèle brut). L'information
apportée par les deux variables est redondante : le modèle ne peut plus identifier la part
de variance qui revient à chacune. Les estimateurs deviennent alors très sensibles à la
moindre modification de l'échantillon (changement de signe possible), et leurs variances
sont artificiellement gonflées.

Concrètement, notre matrice de corrélation montre `r(PRICEEACH, MSRP) = 0,6706`, avec des
VIF de **1,817 et 1,818** : le prix conseillé et le prix pratiqué mesurent largement la
même réalité tarifaire. Interpréter « +1 € de MSRP ⇒ +15,52 € de SALES, toutes choses égales
par ailleurs » est une contre-facture presque irréalisable, puisqu'on ne peut pas faire
varier le prix conseillé en maintenant le prix pratiqué constant. L'inspection de la
matrice de corrélation (et, mieux, des VIF) doit donc précéder **toute** lecture de
coefficient : elle indique si les effets individuels sont identifiables. Le cas échéant, on
supprime une variable redondante, on régularise (ridge) ou on passe à l'ACP/PCR — c'est
précisément le basculement de l'Analyste 2.

### 1.2 Pourquoi un coefficient de détermination (R²) élevé à près de 80 % ne garantit-il pas à lui seul que le modèle de régression est de bonne qualité pour le pilotage commercial ?

Le R² mesure uniquement la part de variance expliquée **dans l'échantillon et avec les
régresseurs retenus** ; il ne teste ni la causalité, ni la stabilité des coefficients, ni
la validité des hypothèses, ni la capacité prédictive hors échantillon. Ici `R² = 0,796` et
`F = 3 665` (p ≈ 0) : le test de Fisher dit seulement que le modèle est meilleur que la
moyenne constante — pas que les élasticités sont pilotables.

Trois raisons supplémentaires, propres à ces données, interdisent de s'en réjouir :

* **Un R² mécanique.** `SALES` est en grande partie une identité comptable : `SALES ≈
  QUANTITYORDERED × PRICEEACH × facteur de majoration` (le facteur vaut 1 dans 53,8 % des
  lignes). Expliquer un montant par les deux quantités qui le composent est presque une
  tautologie : un R² élevé était attendu avant même l'estimation.
* **Des diagnostics effondrés.** `Jarque-Bera = 10 245` (p ≈ 0) et `Durbin-Watson = 1,578`
  montrent que les résidus ne sont ni gaussiens ni indépendants.
* **L'endogénéité.** Le prix est souvent fixé *en réponse* aux ventes attendues : la
  corrélation ne prouve pas le sens de la causalité. Un R² ne corrige pas ce biais.

En résumé : le R² est un indicateur d'ajustement, pas un certificat de décision. Le verdict
doit toujours être rendu après examen des coefficients, des VIF et des résidus.

### 1.3 Pourquoi la présence d'une multicolinéarité non corrigée entre les variables explicatives fausse-t-elle l'interprétation individuelle des tests de Student, malgré des p-valeurs apparentes très faibles ?

La variance d'un estimateur OLS s'écrit
`Var(β̂ⱼ) = σ² / [(1 − R²ⱼ) · Σ(xⱼ − x̄ⱼ)²]`, où `R²ⱼ` est le R² de la régression du
régresseur `j` sur **tous les autres**. Quand `R²ⱼ` tend vers 1, le dénominateur s'effondre :
l'écart-type de `β̂ⱼ` grimpe. Le `t = β̂ⱼ / SE(β̂ⱼ)` devient alors instable — il peut être
artificiellement grand (si le coefficient compense un autre) ou artificiellement petit.

Les p-valeurs « 0.000 » du modèle brut (**37,3** pour PRICEEACH, **29,5** pour MSRP) ne
doivent donc pas être lues comme « chaque variable est indispensable ». Elles testent
`H₀ : βⱼ = 0` *conditionnellement aux autres régresseurs*, une question mal posée quand ces
derniers sont presque redondants. On observe typiquement un F global très significatif
accompagné de t individuels peu fiables ; ici les t sont tous grands, mais les coefficients
changeraient fortement si on retirait une seule ligne ou une seule variable. La conclusion
opérationnelle est la même que dans le PDF : refuser de traduire ces élasticités en
politique de prix tant que la colinéarité n'est pas traitée (log-linéaire, ACP/PCR ou ridge).

### 1.4 Pourquoi l'analyse des résidus du modèle (via des tests comme Jarque-Bera ou Durbin-Watson) est-elle indispensable pour valider ou rejeter la fiabilité des intervalles de confiance calculés ?

Les intervalles de confiance et les tests de Student que `statsmodels` affiche reposent sur
les hypothèses de Gauss-Markov **et**, pour les inférences en échantillon fini, sur
`ε ~ N(0, σ²I)` : normalité, homoscédasticité et indépendance des erreurs. Le logiciel les
calcule automatiquement, que les hypothèses soient vraies ou non : à lui seul, son output
ne dit rien de leur validité.

* **Jarque-Bera = 10 245,7** (p ≈ 0 ; skew = 1,954 ; kurtosis = 11,475) rejette la
  normalité : les erreurs ont des queues très lourdes. Les intervalles `β̂ ± 1,96·SE` sont
  alors mal calibrés, en particulier pour les prévisions et les risques de queue.
* **Durbin-Watson = 1,578** (contre 2 attendu) signale une autocorrélation positive : la
  variance des estimateurs est sous-estimée, les `t` sont gonflés et les intervalles trop
  étroits. Ici, l'ordre des lignes suit les commandes dans le temps, donc la persistance
  saisonnière se retrouve dans les résidus.

Avec n = 2 823, la normalité asymptotique protège partiellement les coefficients, mais pas
la qualité des intervalles de prévision, et elle ne justifie pas l'absence d'autocorrélation.
La conduite à tenir est documentée dans le script : errors robustes (HC3), bootstrap, modèle
log-linéaire (R² = 0,906 ici) ou régression sur composantes principales.

---

## 2. Module Factoriel : l'Analyse en Composantes Principales (ACP)

### 2.1 Pourquoi est-il obligatoire d'appliquer une standardisation (centrage-réduction) aux variables quantitatives avant de lancer une ACP ?

L'ACP recherche les directions de **variance maximale** ; la variance est une grandeur
dépendante de l'unité de mesure. Dans nos données, `SALES` a un écart-type de **1 841,9**
et `QUANTITYORDERED` de **9,7** : sans standardisation, la première composante serait
presque exclusivement « SALES » en euros, non parce que c'est la structure la plus
informative, mais parce que ses chiffres sont les plus grands. `PRICEEACH` (26,88–100) et
`MSRP` (33–214) seraient écrasés.

Le centrage place le nuage sur sa barycentre (les composantes ont une moyenne nulle) et la
réduction donne à chaque variable une variance unitaire. On travaille alors sur la
**matrice de corrélation**, dont la trace vaut exactement `p = 4` : chaque variable pèse
1, les valeurs propres sont comparables, et le critère de Kaiser (λ > 1) devient
interprétable. Autrement dit, la standardisation n'est pas un confort esthétique : elle
définit la métrique même de l'analyse.

### 2.2 Pourquoi l'ACP transforme-t-elle des variables initiales corrélées en de nouvelles composantes principales qui possèdent une corrélation strictement nulle entre elles ?

L'ACP est une diagonalisation de la matrice de corrélation symétrique `R` :
`R vⱼ = λⱼ vⱼ`. Comme `R` est symétrique, ses vecteurs propres forment une **base
orthonormée** (`V'V = I`). Les scores valent `F = Xstandardisé · V`, et leur matrice de
covariance est `F'F/(n−1) = V' R V = diag(λ₁, λ₂, …)` : tous les termes hors diagonale
sont nuls par construction algébrique. Géométriquement, on a simplement tourné le repère
pour aligner les axes avec les directions principales de l'ellipsoïde de dispersion ; les
axes sont perpendiculaires, donc `Cov(PC₁, PC₂) = 0`.

Les corrélations initiales ne disparaissent pas : elles sont **réorganisées** dans les
valeurs propres (l'inertie) et dans les loadings (les corrélations entre variables et
axes). C'est cette orthogonalité qui permet de réinjecter les scores dans une régression
sans multicolinéarité : notre PCR affiche des VIF = 1,0 contre 1,82 pour les prix bruts.

### 2.3 Pourquoi utilise-t-on le critère de Kaiser (valeur propre strictement supérieure à 1) et le pourcentage d'inertie cumulée pour décider du nombre d'axes factoriels à retenir dans l'analyse ?

Les deux critères répondent à deux questions complémentaires.

* **Kaiser (λ > 1)** : après standardisation, chaque variable initiale porte exactement
  1 unité de variance. Un axe dont `λ > 1` contient donc plus d'information qu'une seule
  variable brute : c'est un seuil naturel, indépendant de l'échelle et immédiatement lisible.
  Ici `λ₁ = 2,393` et `λ₂ = 1,153` passent ; `λ₃ = 0,331` et `λ₄ = 0,123` sont écartés.
* **Inertie cumulée** : elle vérifie que la compression reste fidèle (usuellement 70–80 %
  minimum). Notre plan 1–2 restitue **59,82 % + 28,84 % = 88,66 %** de l'inertie totale :
  la perte d'information n'est que de 11,34 %.

Les deux ensemble évitent deux erreurs symétriques : conserver des axes de bruit
(sur-apprentissage, instabilité des coefficients) ou sacrifier une structure réelle pour
un gain de simplicité. Le critère de Kaiser seul peut retenir trop d'axes quand `p` est
grand ; le cumul seul peut retenir un axe marginal. En pratique on les croise avec un test
de stabilité (bootstrap, validation croisée) et avec l'interprétabilité des axes.

*Remarque de rigueur dans le script :* l'atelier calcule ses PC sur les 4 variables **y
compris**, puis régresse `SALES` sur ces PC — fuite de la cible. Nous avons donc refait la
PCR sur les 3 seuls prédicteurs (`λ₁ = 1,671`, `λ₂ ≈ 1`, cumul 89,02 %) : le `R² = 0,794`
est identique à celui du PDF, mais sans leakage.

### 2.4 Pourquoi la lecture simultanée du cercle des corrélations (pour les variables) et du plan factoriel (pour les transactions) permet-elle au décideur de segmenter visuellement ses profils clients sans biais de multicolinéarité ?

Le cercle des corrélations donne la **structure** : la position d'une variable sur un axe
est sa corrélation avec cet axe (son loading), et la longueur de sa flèche mesure la
qualité de représentation (`cos²`). Le plan des individus donne les **coordonnées** : où
se projette chaque transaction. La lecture conjointe (principe d'approximation conjointe
de Benzécri) permet de dire : « ce client est à droite parce que les flèches prix et
facturation pointent dans cette direction ; cet autre est en haut parce que la flèche
volume y règne ».

Dans notre biplot, `PRICEEACH`, `MSRP` et `SALES` se regroupent sur PC1 (59,82 %) — la
« pression tarifaire » — tandis que `QUANTITYORDERED` s'élève sur PC2 (28,84 %) — le
« volume physique ». L'angle entre les deux blocs est proche de 90°, ce qui matérialise
l'indépendance observée (`r = 0,0056`). Comme les axes sont orthogonaux, chaque direction
porte une information distincte : le décideur peut séparer les ventes « forte valeur
unitaire » des ventes « gros volumes » sans double compter le même signal tarifaire.
Précaution indispensable : ne commenter que les points et les variables dont le `cos²` est
élevé, sinon la lecture repose sur une projection trompeuse.

---

## 3. Module Qualitatif : l'Analyse des Correspondances Multiples (ACM)

### 3.1 Pourquoi ne peut-on pas appliquer une ACP classique sur des variables qualitatives (comme les gammes de produits ou les zones géographiques) et pourquoi doit-on passer par un Tableau Disjonctif Complet ?

L'ACP suppose des variables numériques et une géométrie euclidienne : elle calcule des
écarts à la moyenne et des covariance. Or « Classic Cars », « Ships » ou « EMEA » n'ont ni
ordre ni distance entre eux : les coder 1, 2, 3… imposerait une hiérarchie arbitraire
(« Ships » serait deux fois « Classic Cars ») et produirait des composantes dépourvues de
sens.

Le **Tableau Disjonctif Complet** résout le problème en représentant chaque modalité par une
colonne binaire 0/1 : chaque individu a exactement `p = 1` dans chaque bloc de variables
(ici 4 variables, `Q = 18` modalités, matrice 2 823 × 18). On peut alors définir une
distance du χ² entre les profils : chaque modalité est pondérée par l'inverse de son
effectif marginal (`Dc⁻¹`), de sorte qu'une modalité rare ne soit pas écrasée par une
modalité très fréquente comme `Medium` (49,03 %) ou `EMEA` (49,84 %). C'est cette métrique
du χ² — et non l'Euclide — qui est adaptée aux données catégorielles (Greenacre, 1984).

### 3.2 Pourquoi les taux d'inertie brute affichés par les premiers axes d'une ACM sont-ils structurellement très faibles, nécessitant l'application d'un redressement (comme la correction de Benzécri) ?

Dans un tableau disjonctif, chaque ligne est extrêmement creuse : elle ne contient que
`p = 4` valeurs à 1 sur `Q = 18` colonnes. La valeur propre moyenne de la matrice
standardisée vaut donc `1/p = 0,25`, et la part d'inertie « de bruit » est mécaniquement
très importante. Résultat : l'énergie est étalée presque uniformément sur les 14 axes
(`Q − p`), d'où un éboulis plat — ici **8,65 %, 8,37 %, 7,92 %, 7,68 %…** Un lecteur naïf
conclurait à tort qu'aucun axe n'est informatif.

La correction de Benzécri (1979) redresse les valeurs propres au-dessus du seuil
théorique `1/p` : `λ' = [(p/(p−1))(λ − 1/p)]²`, les autres étant mises à zéro. Elle retire
l'inertie artificielle liée à la dimensionalité et révèle le « saut » informationnel :
**45,22 %** sur l'axe 1, **30,14 %** sur l'axe 2, soit **75,36 %** cumulés (au lieu de
17,02 % bruts). Le plan factoriel devient alors lisible et justifie le choix de deux axes.

### 3.3 Pourquoi calcule-t-on les contributions relatives des modalités pour identifier les « moteurs » d'un axe factoriel, plutôt que de se fier uniquement à l'effet de mode des effectifs les plus fréquents ?

La contribution d'une modalité `j` à l'axe `k` vaut `cⱼ · gⱼ² / λₖ` : elle combine son
poids marginal `cⱼ` **et** son éloignement `gⱼ` du centre du nuage. Une modalité très
fréquente mais proche du centre (par exemple `Medium`, 49,03 % des commandes) contribue
peu ; une modalité plus rare mais très excentrée peut être déterminante. Se fier aux
effectifs reviendrait à confondre « populaire » et « structurant ».

Le seuil de référence est `100/Q = 5,56 %`. Nos résultats montrent que l'axe 1 est porté
par `PRODUCTLINE_Classic Cars` (**24,02 %**, `cos² = 0,528`), `DEALSIZE_Large` (**21,40 %**)
et `DEALSIZE_Small` (**18,41 %**) — soit 63,83 % à eux trois —, ce qui corrobore la lecture
« valeur commerciale ». L'axe 2 est piloté par `TERRITORY_APAC` (17,30 %),
`TERRITORY_EMEA` (15,38 %), `TERRITORY_NA_Americas` (11,04 %) et `TRIMESTRE_T3` (10,78 %) :
c'est la segmentation spatio-temporelle.

En complément, le `cos²` vérifie que la modalité est bien représentée dans le plan : on
retient une modalité comme « moteur » seulement si sa contribution est élevée **et** sa
qualité de représentation correcte.

### 3.4 Pourquoi associe-t-on souvent la géométrie de l'ACM à une Classification Ascendante Hiérarchique (CAH) pour aider le management à structurer des plans d'action ciblés par segments de clientèle (Husson et al., 2017) ?

L'ACM fournit des **continuum de coordonnées** et des proximités : elle dit que
`Classic Cars` et `Large` sont proches, mais elle ne découpe pas elle-même le marché en
groupes opérationnels. La CAH (ici Ward sur les 5 premières dimensions factorielles)
transforme ce nuage en **hiérarchie de classes**, avec un dendrogramme lisible et un choix
de niveau de coupe (4 groupes dans notre script). On obtient des segments nommables et
directement actionnables (« grands comptes Classic Cars », « gammes secondaires petits
contrats », « profils APAC/T3 », etc.).

Trois raisons rendent le couplage particulièrement pertinent :

1. **Débruitage** : la CAH travaille sur les axes factoriels, donc sur le signal déjà
   filtré par l'ACM et corrigé par Benzécri, pas sur les 18 colonnes brutes redondantes ;
2. **Lisibilité managériale** : l'arbre montre les emboîtements (un segment général se
   subdivision en sous-segments), ce qui aide à construire une politique « global → ciblé » ;
3. **Décision** : chaque classe reçoit une action distincte (Key Account, promotions par
   paliers, priorité logistique), ce qui est exactement la demande du PDG : passer d'une
   cartographie à des plans d'action chiffrés.

---

## 4. Module Prédictif : les Machines à Vecteurs de Support (SVM)

### 4.1 Pourquoi la standardisation des variables explicatives est-elle une étape critique et non négociable avant d'entraîner un algorithme de classification par SVM (Hastie et al., 2009) ?

Le SVM optimise une **marge euclidienne** et utilise un noyau qui dépend de distances
`‖xᵢ − xⱼ‖`. Si `SALES` a un écart-type de 1 842 et `QUANTITYORDERED` de 9,7, la distance
est dominée par le chiffre d'affaires : le noyau RBF « voit » les autres variables comme
du bruit, et les hyperparamètres `C` et `γ` perdent leur sens (γ est défini par
`1/(p·Var(X))` avec `gamma='scale'`). Le centrage-réduction place toutes les features sur
la même échelle, garantissant que la pénalité et la largeur de marge s'appliquent
équitablement.

La règle d'or est d'ajuster le `StandardScaler` **sur l'ensemble d'entraînement
uniquement** (`fit(X_train)`), puis de ne faire que `transform(X_test)` : ajuster sur le
jeu de test ferait fuir la moyenne et l'écart-type du test dans l'entraînement (data
leakage) et donnerait une performance optimiste.

*Observation honnête sur nos données :* un SVM brut non standardisé atteint aussi 100 %
ici, parce que `SALES` suffit presque à lui seul à reconstituer `DEALSIZE`. Ce résultat
est un **artefact de ce jeu de données** ; il ne dispense pas de la standardisation, qui
reste obligatoire dès que les features changent d'échelle ou que de nouvelles variables
sont ajoutées.

### 4.2 Pourquoi l'utilisation d'un noyau non linéaire (comme le noyau RBF) permet-elle de séparer des classes de transactions complexes là où une droite linéaire classique échouerait ?

Le noyau RBF, `K(x, x') = exp(−γ‖x − x'‖²)`, projette implicitement les observations dans
un espace de dimension infinie (kernel trick) où un **hyperplan linéaire** correspond à une
frontière **courbe** dans l'espace d'origine. Le paramètre `γ` règle la portée de
l'influence de chaque point : petit γ ⇒ frontière lisse, grand γ ⇒ frontière très
sinueuse (risque de sur-apprentissage), tandis que `C` arbitre entre marge large et
erreurs de classification. C'est ce mécanisme qui permet d'épouser des nuages en forme de
poches, d'ellipses imbriquées ou de zones disjointes qu'un simple `w·x + b = 0` ne peut pas
séparer.

Notre figure `figures/fig_05_svm_frontiere_roc.png` trace cette frontière non linéaire
après projection ACP 2D. **Mais l'esprit critique impose une réserve** : dans ce jeu précis,
le SVM **linéaire** obtient `accuracy = 1,000` et `F1 = 1,000`, donc la supériorité du RBF
n'est pas démontrée ici. Le noyau RBF reste un choix prudent par défaut pour des
frontières potentiellement complexes, mais il doit être validé par comparaison croisée
(notre RBF : AUC CV = 0,9996 ± 0,0004), et non postulé comme une vérité.

### 4.3 Pourquoi évalue-t-on la performance d'un modèle de classification prédictif à l'aide d'indicateurs comme l'Aire sous la courbe ROC (AUC-ROC) et la matrice de confusion, plutôt que de se limiter à la simple précision globale (Accuracy) lorsque la classe cible est rare ?

Parce que l'accuracy est **dominée par la classe majoritaire**. Ici, 157 commandes seulement
sont `Large` sur 2 823 (**5,56 %**) : un classifieur idiot qui prédit toujours « Non-Large »
obtient une accuracy de **94,48 %** — presque aussi « bien » que notre SVM (99,58 %) — tout
en ratant 100 % des grands comptes. L'accuracy seule masque exactement l'erreur qui coûte
cher à l'entreprise.

* **La matrice de confusion** (ici 666 TN, 1 FP, 2 FN, 37 TP) montre *quelles* erreurs
  sont faites : 2 grands comptes manqués et 1 fausse alerte. Elle donne accès au rappel
  (`37/39 = 94,87 %`), à la précision (`37/38 = 97,37 %`) et au `F1 = 0,961` de la classe
  critique — des indicateurs alignés sur l'objectif métier.
* **L'AUC-ROC = 0,9999** mesure la qualité du classement des cas par score de probabilité,
  indépendamment du seuil de décision et déséquilibré : c'est le bon critère pour choisir
  later un seuil adapté au coût respectif des faux positifs et des faux négatifs.

La règle pratique est donc : accuracy pour le bilan global, matrice de confusion + F1 pour
la classe rare, AUC-ROC pour comparer des modèles et régler le seuil.

### 4.4 Pourquoi un modèle SVM performant à 99 % sur des données historiques constitue-t-il un levier de rentabilité opérationnelle directe pour automatiser le scoring des commandes dans un système CRM (Sztompka / Pareto) ?

Parce qu'il transforme une observation rétrospective en **décision exécutable au moment où
la commande arrive**. Intégré à l'ERP/CRM, le modèle calcule une probabilité « Grand Compte »
et déclenche des règles métier : (A) lead scoring en temps réel avec signalement des
commandes à fort potentiel, (B) routage vers le Key Account Management et priorisation
logistique, (C) offres incitatives pour les profils « proches du seuil », (D) automatisation
complète de la facturation/expédition des commandes classées Non-Large avec forte certitude
— soit moins de frais administratifs sur leiseur du volume. C'est exactement la logique
de Pareto/Sztompka : concentrer ~80 % de l'attention sur les ~5,6 % de transactions qui
portent la valeur, et automatiser la longue traîne.

Deux réserves décisionnelles doivent accompagner ce levier, et le script les documente :

1. **Circularité partielle** : `DEALSIZE` est une discrétisation de `SALES` (`Small < 3000`,
   `Medium` entre 3000 et 7000, `Large > 7000`). Avec `SALES` parmi les features, le SVM
   redécouvre une règle métier déjà connue. Sans `SALES`, la performance tombe à
   `accuracy = 97,31 %`, `AUC = 0,9719`, `F1 = 0,716` — encore utile, mais plus modeste.
   Pour un vrai scoring *anticipatif*, il faut n'utiliser que des variables connues **avant**
   la commande (historique client, panier, canal, saison).
2. **Gouvernance** : surveiller la dérive du modèle, recalibrer le seuil de probabilité selon
   le coût d'un faux négatif et garder une revue humaine sur les cas frontières. Un score à
   99 % sur l'historique n'est pas une garantie sur le futur.

**Verdict :** le SVM est la méthode retenue pour la décision finale, mais présentée au DG
comme un outil d'aide à la décision — alimenté par un nettoyage rigoureux, éclairé par
l'ACP et l'ACM, et assorti de ses limites — et non comme une vérité automatique.

---

## Synthèse finale

| Module | Méthode retenue | Résultat clé |
|---|---|---|
| MCO | OLS + diagnostics + correctif log | `R² = 0,796`, JB = 10 245, DW = 1,578 → élasticités brutes rejetées |
| ACP | Standardisation + 2 axes + PCR sans leakage | `λ₁ = 2,393`, `λ₂ = 1,153`, 88,66 % d'inertie, VIF = 1 |
| ACM | χ² + Benzécri + CAH | 45,22 % + 30,14 % = 75,36 % corrigé ; 4 segments |
| SVM | RBF, C = 10, γ = scale, standardisation train-only | Accuracy 99,58 %, AUC 0,9999, F1 0,961 (classe Large) |

Tous ces chiffres sont reproductibles en local avec :

```bash
cd /home/becem69/datasc-final
python3 analyse_complete.py
```
