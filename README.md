# 📝 Time Series Papers
A comprehensive survey on the time series papers from 2018-2022 (we will update it in time ASAP!) on the top conferences (NeurIPS, ICML, ICLR, SIGKDD, SIGIR, AAAI, IJCAI, WWW, CIKM, ICDM, WSDM, etc.)

We divided these papers into several fundamental tasks as follows.
# Time Series Dataset
A comprehensive time-series dataset survey.

These datasets are classified based on the following tasks.
- [📝 Time Series Papers](#-time-series-paper)
  - [Survey](#survey)
  - [Time Series Forecasting](#time-series-forecasting)
  - [Time Series Classification ](#time-series-classification)
  - [Anomaly Detection ](#anomaly-detection)
  - [Time series Clustering ](#time-series-clustering)
  - [Time Series Segmentation](#time-series-segmentation)
  - [Others ](#others)


## Survey

|             Paper                                                           | Conference | Year | Code | Used Datasets |Key Contribution|
| :--------------------------: | :-------------------: | :------------------: | ----------------------- | ----------------------- |------ |
|[Transformers in Time Series: A Survey](https://arxiv.org/pdf/2202.07125.pdf)| - | 2022 | [link](https://github.com/qingsongedu/time-series-transformers-review) | - |
|[Time series data augmentation for deep learning: a survey](https://arxiv.org/pdf/2002.12478.pdf)| IJCAI | 2021 | - | - |
|[Neural temporal point processes: a review](https://arxiv.org/pdf/2104.03528v5.pdf)| IJCAI | 2021 | - | - |
|[Time-series forecasting with deep learning: a survey](https://arxiv.org/pdf/2004.13408.pdf)| Philosophical Transactions of the Royal Society A | 2021 | - | - |
|[Deep learning for time series forecasting: a survey](https://www.liebertpub.com/doi/10.1089/big.2020.0159)| Big Data | 2021 | - | - |
|[DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction](https://arxiv.org/pdf/2108.09091.pdf)| CIKM | 2021 | [graph-data](https://github.com/deepkashiwa20/DL-Traff-Graph), [grid-data](https://github.com/deepkashiwa20/DL-Traff-Grid) | - |
|[Graph Neural Network for Traffic Forecasting: A Survey](https://arxiv.org/pdf/2101.11174v3.pdf)| - | 2021 | - | - |
|[Deep learning for anomaly detection in time-series data: review, analysis, and guidelines](https://ieeexplore.ieee.org/abstract/document/9523565)| Access | 2021 | - | - |
|[A review on outlier/anomaly detection in time series data](https://arxiv.org/pdf/2002.04236.pdf)| ACM Computing Surveys | 2021 | - | - |
|[A unifying review of deep and shallow anomaly detection](http://128.84.4.34/pdf/2009.11732)| Proceedings of the IEEE | 2021 | - | - |
|[Big Data for Traffic Estimation and Prediction: A Survey of Data and Tools](https://www.mdpi.com/2571-5577/5/1/23)| Applied System Innovation 5 | 2021 | - | - |
|[Fusion in stock market prediction: A decade survey on the necessity, recent developments, and potential future directions](https://www.sciencedirect.com/science/article/pii/S1566253520303481)| Information Fusion | 2021 | - | - |
|[Applications of deep learning in stock market prediction: Recent progress](https://www.sciencedirect.com/science/article/pii/S0957417421009441)| ESA | 2021 | - | - |
|[Deep Learning for Spatio-Temporal Data Mining: A Survey](https://ieeexplore.ieee.org/abstract/document/9204396)| KDD | 2020 | - | - |
|[Urban flow prediction from spatiotemporal data using machine learning: A survey ](https://www.sciencedirect.com/science/article/abs/pii/S1566253519303094)| Information Fusion | 2020 | - | - |
|[An empirical survey of data augmentation for time series classification with neural networks](https://arxiv.org/pdf/2007.15951.pdf)| - | 2020 | [link](https://github.com/uchidalab/time_series_augmentation) | - |
|[Deep Learning on Traffic Prediction: Methods, Analysis and Future Directions](https://arxiv.org/pdf/2004.08555.pdf)| - | 2020 | - | - |
|[Neural forecasting: Introduction and literature overview](https://arxiv.org/pdf/2004.10240.pdf)| - | 2020 | - | - |
|[Deep Learning on Traffic Prediction: Methods, Analysis and Future Directions](https://arxiv.org/pdf/2004.08555.pdf)| - | 2020 | - | - |
|[Deep learning for time series classification: a review](https://arxiv.org/pdf/2004.08555.pdf)| Data Mining and Knowledge Discovery | 2019 | [link](https://github.com/hfawaz/dl-4-tsc) | - |
|[Financial time series forecasting with deep learning : A systematic literature review: 2005–2019](https://arxiv.org/pdf/1911.13288.pdf)| ASC | 2019 | - | - |
|[Natural language based financial forecasting: a survey](https://dspace.mit.edu/bitstream/handle/1721.1/116314/10462_2017_9588_ReferencePDF.pdf?sequence=2&isAllowed=y)| Artificial Intelligence Review | 2018 | - | - |




## Time Series Forecasting 

|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
|[CATN: Cross Attentive Tree-aware Network for Multivariate Time Series Forecasting](https://www.aaai.org/AAAI22Papers/AAAI-7403.HeH.pdf)| AAAI | 2022 | - | [Traffic](https://archive.ics.uci.edu/ml/datasets/PEMS-SF), [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [PeMSD7(M)](https://dot.ca.gov/programs/traffic-operations/mpr/pemssource), [METR-LA](http://ceur-ws.org/Vol-2750/paper9.pdf) |
| [Reinforcement Learning based Dynamic Model Combination for Time Series Forecasting](https://arxiv.org/abs/2022.01846)| AAAI | 2022 | - | [DATA](https://nsrdb.nrel.gov) |
|[Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](https://arxiv.org/abs/2101.01000)|AAAI | 2022 | [Code link](https://github.com/BIRD-TAO/CLCRN) | - |
| [TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs](https://arxiv.org/abs/2112.08025) | AAAI | 2022 | [Code link](https://github.com/liu-yushan/TLogic) | - |
| [Spatio-Temporal Recurrent Networks for Event-Based Optical Flow Estimation](https://arxiv.org/abs/2109.04871) | AAAI | 2022 | - | - |
| [A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction](https://arxiv.org/pdf/2111.08900.pdf) | AAAI | 2022 | - | - |



## Time Series Classification 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |



## Anomaly Detection

|             Dataset             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :------------------------: | ----------------------- | ------------------------- |------ |
| [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/abs/2109.05257) | AAAI | 2022 | - | - |
| [DeepGPD: A Deep Learning Approach for Modeling Geospatio-Temporal Extreme Events]() | AAAI | 2022 | - | - |



## Time series Clustering 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Clustering Interval-Censored Time-Series for Disease Phenotyping](https://arxiv.org/abs/2102.07005v4) | AAAI | 2022 | - | - |

## Time series Segmentation 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |


## Others

|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Conditional Loss and Deep Euler Scheme for Time Series Generation](https://arxiv.org/abs/2102.05313v5) | AAAI | 2022 | - | - |

