# Catching the High Frequency Traders

﻿Financial markets are made up of various types of market players, each with specific interests and hence heterogeneous behaviors. For regulators, it is important to know the different types of market players in order to better understand how their behavior has an impact on the market. Since the emergence of **High-Frequency Trading** (HFT) more than a decade ago, financial sector authorities as well as academics have widely studied the impact and influence on markets of these market players [1].

Some professionals criticize high-frequency trading since they believe that it gives an **unfair advantage** to large firms and unbalances the playing field. It can also harm other investors that hold a long-term strategy and buy or sell in bulk.

This project was developed in the context of a data challenge organized by **ENS** and **l'autorité des marchés financiers** ([AMF](https://www.amf-france.org/)). The goal was to **identify** and **classify** <u>traders</u> within <u>three categories</u>, **HFT**, **non HFT** and **MIX**.

[[1]](https://challengedata.ens.fr/participants/challenges/50/) **AMF Data Challenge**

## Our approach

We identify important **indicators**, known as **features**, thanks to **PCA** and gradient boosting analysis. Then, we train multiple models such as **MLP**, **Decision Trees**, and **Gradient Boosting** on 10/18 features that were previously deemed relevant for our classification problem.

**We achieved <u>first place</u> in the private leaderboard with an accuracy of <u>100%</u>**

## Authors

[Jean-Charles LAYOUN](https://www.linkedin.com/in/jclayoun). You can get in touch with me at [jean-charles.layoun@polytechnique.edu](mailto:jean-charles.layoun@polytechnique.edu).

[Ronan Sangouard](https://www.linkedin.com/in/ronan-sangouard-516766193). You can get in touch with me at [ronan.sangouard@polytechnique.edu](mailto:ronan.sangouard@polytechnique.edu).
