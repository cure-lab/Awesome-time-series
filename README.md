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
|[CATN: Cross Attentive Tree-aware Network for Multivariate Time Series Forecasting](https://www.aaai.org/AAAI22Papers/AAAI-7403.HeH.pdf)| AAAI | 2022 | - | [Traffic](https://archive.ics.uci.edu/ml/datasets/PEMS-SF), [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [PeMSD7(M)](https://dot.ca.gov/programs/traffic-operations/mpr/pemssource), [METR-LA](http://ceur-ws.org/Vol-2750/paper9.pdf) |  studied the hierarchical and grouped correlation mining problem of multivariate time-series data and proposed CATN for multi-step forecasting. |
| [Reinforcement Learning based Dynamic Model Combination for Time Series Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9564132)| AAAI | 2022 | - | [DATA](https://nsrdb.nrel.gov) | a novel and practically effective online ensemble aggregation framework for time-series forecasting that employs a deep reinforcement learning approach as a meta-learning technique. |
|[Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](https://arxiv.org/abs/2101.01000)|AAAI | 2022 | [Code link](https://github.com/BIRD-TAO/CLCRN) | WeatherBench (Rasp et al. 2020) | a local conditional convolution to capture and imitate the meteorological flows of local patterns on the whole sphere|
| [TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs](https://arxiv.org/abs/2112.08025) | AAAI | 2022 | [Code link](https://github.com/liu-yushan/TLogic) | [ Integrated Cri- sis Early Warning System](https://dataverse.harvard.edu/dataverse/icews), [Split method](https://github.com/TemporalKGTeam/xERTE) |the first symbolic framework that directly learns temporal logical rules from temporal knowl- edge graphs and applies these rules for link forecasting|
| [Spatio-Temporal Recurrent Networks for Event-Based Optical Flow Estimation](https://arxiv.org/abs/2109.04871) | AAAI | 2022 | - | The MVSEC dataset (Zhu et al. 2018a) | novel input representation to effectively extract the spatio-temporal information from event input.｜
| [A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction](https://arxiv.org/pdf/2111.08900.pdf) | AAAI | 2022 | - | Crop |  a novel GNN-RNN framework to innovatively incorporate both geospatial and temporal knowledge into crop yield prediction.|
| [Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting](https://arxiv.org/abs/2009.05135) | AAAI | 2021 | - | [Pacific Ocean Temperature Dataset](http://iridl.ldeo.columbia.edu/), [Parking Birmingham Data Set](https://data.birmingham.gov.uk/dataset/birmingham-parking), ........ | it parameterizes the weights in terms of a deep switching vector auto-regressive likelihood governed with a Markovian prior |
| [Dynamic Gaussian Mixture Based Deep Generative Model for Robust Forecasting on Sparse Multivariate Time Series](https://arxiv.org/abs/2103.02164) | AAAI | 2021 | - | [USHCN](https://www.ncdc.noaa.gov/ushcn/introduction), [KDD-CUP](https://www.kdd.org/kdd2018/kdd-cup), MIMIC-III |provides a novel and general solution that explicitly defines temporal dependency between Gaussian mixture distributions at different time steps |
| [Temporal Latent Autoencoder: A Method for Probabilistic Multivariate Time Series Forecasting](https://www.aaai.org/AAAI21Papers/AAAI-3796.NguyenN.pdf) | AAAI | 2021 | -| Traffic, Electricity, Wiki | introduced a novel temporal latent auto-encoder method which enables nonlinear factorization of multivariate time series, learned end-to-end with a temporal deep learning latent space forecast model. By imposing a probabilistic latent space model, complex distributions of the input series are modeled via the decoder.|
| [Synergetic Learning of Heterogeneous Temporal Sequences for Multi-Horizon Probabilistic Forecasting](https://arxiv.org/abs/2102.00431) | AAAI | 2021 | - | Electricity(UCI), Traffic, Environment(Li, L.;Yan,J.;Yang,X.;and Jin,Y.2019a.) | presented a novel approach based on the deep conditional generative model to jointly learn from heterogeneous temporal sequences.|



## Time Series Classification 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Correlative Channel-Aware Fusion for Multi-View Time Series Classification](https://arxiv.org/abs/1911.11561) | AAAI | 2021 | - | EV-Action, NTU RGB+D, UCI Daily and Sports Activities | The global-local temporal encoders are developed to extract robust temporal representations for each view, and a learnable fusion mechanism is proposed to boost the multi-view label information. |
| [Learnable Dynamic Temporal Pooling for Time Series Classification](https://arxiv.org/abs/2104.02577) | AAAI | 2021 | - | UCR/UEA |proposes a dynamic temporal pooling + a learning framework to simultaneously optimize the network parameters of a CNN classifier and the prototypical hidden series that encodes the latent semantic of the segments. |
| [ShapeNet: A Shapelet-Neural Network Approach for Multivariate Time Series Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17018) | AAAI | 2021 | [Code link](http://alturl.com/d26bo) | UEA MTS datasets | We propose Mdc-CNN to learn time series subsequences of various lengths into unified space and propose a cluster-wise triplet loss to train the network in an unsupervised fashion. We adopt MST to obtain the MST representation of time series. |
| [Joint-Label Learning by Dual Augmentation for Time Series Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17071) | AAAI | 2021 | [Code](https://github.com/fchollet/keras) | [UCR](https://www.cs.ucr.edu/˜eamonn/) | a novel time-series data augmentation method  |


## Anomaly Detection

|             Dataset             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :------------------------: | ----------------------- | ------------------------- |------ |
| [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/abs/2109.05257) | AAAI | 2022 | - | Secure water treatment (SWaT), ...... | applying PA can severely overestimate a TAD model’s capability.|
| [DeepGPD: A Deep Learning Approach for Modeling Geospatio-Temporal Extreme Events](https://aaai-2022.virtualchair.net/poster_aaai10861) | AAAI | 2022 | [Code link](https://github.com/TylerPWilson/deepGPD) | [the Global Historical Climatology Network (GHCN)](https://www.ncdc.noaa.gov/ghcn-daily-description) |proposed a novel deep learning architecture (DeepGPD) capable of learning the parameters of the generalized Pareto distribution while satisfying the conditions placed on those parameters.|
| [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series]() | AAAI | 2021 | - | - | - | - |
| [Time Series Anomaly Detection with Multiresolution Ensemble Decoding]() | AAAI | 2021 | - | - | - |- |
| [Outlier Impact Characterization for Time Series Data]() | AAAI | 2021 | - | - | - |- |


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
| [Conditional Loss and Deep Euler Scheme for Time Series Generation](https://arxiv.org/abs/2102.05313v5) | AAAI | 2022 | - | - | - |
| [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) | AAAI | 2022 | [Code link](https://github.com/yuezhihan/ts2vec) | [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/),[30 UEA datasets](http://www.timeseriesclassification.com/), [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset), [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70), [KPI dataset](http://test-10056879.file.myqcloud.com/10056879/test/20180524_78431960010324/KPI%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%86%B3%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86.zip)  | performs contrastive learning in a hierarchical way over augmented context views|
| [Generative Semi-Supervised Learning for Multivariate Time Series Imputation]() | AAAI | 2021 | - | - | - |
