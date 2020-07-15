# Replication Study: Galiani, Rossi, and Schargrodsky (2011) Revisited

I replicate the findings of the following journal article:

Galiani, Sebastian, Martín A. Rossi, and Ernesto Schargrodsky. 2011. "Conscription and Crime: Evidence from the Argentine Draft Lottery." *American Economic Journal: Applied Economics*, 3 (2): 119-36.
[DOI: 10.1257/app.3.2.119](http://dx.doi.org/10.1257/app.3.2.119)

<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/microeconometrics-course-project-bhmueller/blob/master/replication_conscription_crime.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/microeconometrics-course-project-bhmueller/master?filepath=replication_conscription_crime.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>
[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/microeconometrics-course-project-bhmueller.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/microeconometrics-course-project-bhmueller)

## Summary

Galiani et al. (2011) aim at studying the causal effect of serving in the military on subsequent criminal behaviour. They use administrative data on the Argentine draft lottery and criminal records. Since failing medical examination before being drafted into the military is endogenous, Galiani et al. instrument conscription by *draft eligible*, a variable indicating whether a unit of observation was randomly chosen to join military service. The unit of observation are all individuals in birth cohort c sharing the same last three digits of the national ID.
The authors find that the effect of conscription on developing a criminal record is positive and statistically significant.

I replicate the findings of Galiani et al. (2011). Further, I discuss the identification strategy by drawing on the potential outcome model of instrumental variable estimation by Imbens and Angrist (1994) and causal graphs. I extend the authors' analysis by providing additional suggestive evidence of the exogeneity of the instrument.


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/LICENSE)
[![Continuous Integration](https://github.com/HumanCapitalAnalysis/microeconometrics-course-project-bhmueller/workflows/Continuous%20Integration/badge.svg)](https://github.com/HumanCapitalAnalysis/microeconometrics-course-project-bhmueller/actions)
