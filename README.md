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
| [ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction](https://dl.acm.org/doi/10.1145/3488560.3498444) | WSDM | 2022 | [Code link](https://github.com/k51/STGSP) | TaxiBJ, BikeNYC | our model explicitly models the correlation among temporal dependencies of different scales to extract global temporal dependencies + new simple fusion strategy + self-supervised learning |
| [PYRAFORMER: LOW-COMPLEXITY PYRAMIDAL ATTENTION FOR LONG-RANGE TIME SERIES MODELING AND FORECASTING](https://openreview.net/pdf?id=0EXmFzUn5I) | ICLR | 2022 | [Code link](https://github.com/alipay/Pyraformer) | [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [Wind](https://www.kaggle.com/sohier/30-years-of-european-wind-generation), [ETT data](https://github.com/zhouhaoyi/ETDataset) and [App Flow](https://github.com/alipay/Pyraformer/tree/master/data) |  a novel model based on pyramidal attention that can effectively describe both short and long temporal dependencies with low time and space complexity. |
| [DEPTS: DEEP EXPANSION LEARNING FOR PERIODIC TIME SERIES FORECASTING](https://openreview.net/pdf?id=AJAR-JgNw__) | ICLR | 2022 | [Code link](https://github.com/weifantt/DEPTS) | ELECTRICITY, TRAFFIC2, and M4(HOURLY) | - |
| [TAMP-S2GCNETS: COUPLING TIME-AWARE MULTIPERSISTENCE KNOWLEDGE REPRESENTATION WITH SPATIO-SUPRA GRAPH CONVOLUTIONAL NETWORKS FOR TIME-SERIES FORECASTING](https://openreview.net/pdf?id=wv6g8fWLX2q) | ICLR | 2022 | [Code link](https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0.) | PeMSD3, PeMSD4, PeMSD8 and COVID-19 | - |
| [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://arxiv.org/abs/2202.01575) | ICLR | 2022 | [Code link](https://github.com/salesforce/CoST) | [ETT](https://github.com/zhouhaoyi/ETDataset),[Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014),[Weather](https://www.ncei.noaa.gov/data/local-climatological-data/) |- |
| [REVERSIBLE INSTANCE NORMALIZATION FOR ACCURATE TIME-SERIES FORECASTING AGAINST DISTRIBUTION SHIFT](https://openreview.net/pdf?id=cGDAkQo1C0p) | ICLR | 2022 | - | [ETT](https://github.com/zhouhaoyi/ETDataset), [Electricity Consuming Load (ECL)](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |- |
| [TEMPORAL ALIGNMENT PREDICTION FOR SUPERVISED REPRESENTATION LEARNING AND FEW-SHOT SEQUENCE CLASSIFICATION](https://openreview.net/pdf?id=p3DKPQ7uaAi) | ICLR | 2022 | [Code link](https://github.com/BingSu12/TAP) | MSR Action3D, MSR Daily Activity3D, “Spoken Arabic Digits (SAD)” dataset, ChaLearn | - |
| [Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting](https://arxiv.org/abs/2009.05135) | AAAI | 2021 | - | [Pacific Ocean Temperature Dataset](http://iridl.ldeo.columbia.edu/), [Parking Birmingham Data Set](https://data.birmingham.gov.uk/dataset/birmingham-parking), ........ | it parameterizes the weights in terms of a deep switching vector auto-regressive likelihood governed with a Markovian prior |
| [Dynamic Gaussian Mixture Based Deep Generative Model for Robust Forecasting on Sparse Multivariate Time Series](https://arxiv.org/abs/2103.02164) | AAAI | 2021 | - | [USHCN](https://www.ncdc.noaa.gov/ushcn/introduction), [KDD-CUP](https://www.kdd.org/kdd2018/kdd-cup), MIMIC-III |provides a novel and general solution that explicitly defines temporal dependency between Gaussian mixture distributions at different time steps |
| [Temporal Latent Autoencoder: A Method for Probabilistic Multivariate Time Series Forecasting](https://www.aaai.org/AAAI21Papers/AAAI-3796.NguyenN.pdf) | AAAI | 2021 | -| Traffic, Electricity, Wiki | introduced a novel temporal latent auto-encoder method which enables nonlinear factorization of multivariate time series, learned end-to-end with a temporal deep learning latent space forecast model. By imposing a probabilistic latent space model, complex distributions of the input series are modeled via the decoder.|
| [Synergetic Learning of Heterogeneous Temporal Sequences for Multi-Horizon Probabilistic Forecasting](https://arxiv.org/abs/2102.00431) | AAAI | 2021 | - | Electricity(UCI), Traffic, Environment(Li, L.;Yan,J.;Yang,X.;and Jin,Y.2019a.) | presented a novel approach based on the deep conditional generative model to jointly learn from heterogeneous temporal sequences.|
| [Time-Series Event Prediction with Evolutionary State Graph](https://arxiv.org/pdf/1905.05006.pdf) | WSDM | 2021 | [Code link](https://github.com/VachelHU/EvoNet) | DJIA30, WebTraffic, NetFlow, ClockErr, and AbServe | proposed a novel represen- tation, the evolutionary state graph, to present the time-varying re- lations among time-series states. |
| [Long Horizon Forecasting With Temporal Point Processes](https://arxiv.org/pdf/2101.02815.pdf) | WSDM | 2021 | [Code link](https://github.com/pratham16cse/DualTPP) | Election, Taxi, Traffic-911, and EMS-911. | a novel MTPP model specif- ically designed for long-term forecasting of events. |
| [Modeling Inter-station Relationships with Attentive Temporal Graph Convolutional Network for Air Quality Prediction](https://dl.acm.org/doi/pdf/10.1145/3437963.3441731) | WSDM | 2021 | - | [Beijing](https://beijingair.sinaapp.com/), [Tianjin](http://urban-computing.com/data/Data-1.zip) and [POIs data](https://map.baidu.com/)|  encode multiple types of inter-station relationships into graphs and design parallel GCNbased encoding and decoding modules to aggregate features from related stations using different graphs. |
| [Predicting Crowd Flows via Pyramid Dilated Deeper Spatial-temporal Network](https://dl.acm.org/doi/pdf/10.1145/3437963.3441785) | WSDM | 2021 | - |  Wi-Fi connection log,  [bike in New York city](https://ride.citibikenyc.com/system-data) and [taxi ride in New York](www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | ConvLSTM + pyramid dilated residual network + integrated attention |
| [Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting](https://proceedings.mlr.press/v139/chen21o.html) | ICML | 2021 | [Code link](https://github.com/Z-GCNETs/Z-GCNETs.git) | Decentraland, Bytom, PeMSD4 and PeMSD8. | The new Z-GCNETs layer allows us to track the salient timeaware topological characterizations of the data persisting over time. |
| [Explaining Time Series Predictions with Dynamic Masks](https://proceedings.mlr.press/v139/crabbe21a.html) | ICML |2021 |  [Code link](https://github.com/JonathanCrabbe/Dynamask) | MIMIC-III | These masks are endowed with an insightful information theoretic interpretation and offer a neat improvement in terms of performance. |
| [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series](https://proceedings.mlr.press/v139/rangapuram21a.html) |  ICML |2021 | [Code link](https://github.com/awslabs/gluon-ts) | Labour, Traffic, Tourism, Tourism-L, and Wiki | a single, global model that does not require any adjustments to produce coherent, probabilistic forecasts, a first of its kind. |
| [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://proceedings.mlr.press/v139/rasul21a.html) | ICML |2021 | [Code link](https://github.com/zalandoresearch/pytorch-ts) | [Exchange, Solar and Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014),  [Traffic](https://archive.ics.uci.edu/ml/datasets/PEMS-SF), [Taxi](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and [Wikipedia](https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release/datasets) | a combination of improved variance schedule and an L1 loss to allow sampling with fewer steps at the cost of a small reduction in quality if such a trade-off is required. |
| [Conformal prediction interval for dynamic time-series](https://proceedings.mlr.press/v139/xu21h.html) | ICML |2021 |  [Code link](https://github.com/hamrel-cxu/EnbPI) | solar and wind energy data | present a predictive inference method for dynamic time-series. |
| [RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting](https://proceedings.mlr.press/v139/pal21b.html) | ICML |2021 | - | PeMSD3, PeMSD4, PeMSD7 and PeMSD8 | - |
| [ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting](https://dl.acm.org/doi/pdf/10.1145/3447548.3467330) |  KDD |2021 | [Code link](https://github.com/JLDeng/ST-Norm) | BikeNYC, PeMSD7 and Electricity |- |
| [MiniRocket: A Fast (Almost) Deterministic Transform for Time Series Classification](https://dl.acm.org/doi/pdf/10.1145/3447548.3467231) | KDD |2021 |  [Code link](https://github.com/angus924/minirocket) | UCR archive |- |
| [Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting](https://dl.acm.org/doi/pdf/10.1145/3447548.3467275)|KDD|2021|[Code link](https://github.com/liangzhehan/DMSTGCN)| PeMSD4, PeMSD8 and England | - |
| [Forecasting Interaction Order on Temporal Graphs](https://dl.acm.org/doi/pdf/10.1145/3447548.3467341)|KDD|2021| [Code link](https://github.com/xiawenwen49/TAT-code)| COLLEGEMSG, EMAIL-EU and FBWALL |-|
|[Quantifying Uncertainty in Deep Spatiotemporal Forecasting](https://dl.acm.org/doi/pdf/10.1145/3447548.3467325)| KDD | 2021 | - | air quality PM2.5, road network traffic, and COVID-19 incident deaths | - |
|[Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting](https://dI.acm.org/doi/pdf/10.1145/3447548.3467430)|KDD| 2021 |[Code link](https://github.com/square-coder/STGODE)|-|-|
|[A PLAN for Tackling the Locust Crisis in East Africa: Harnessing Spatiotemporal Deep Models for Locust Movement Forecasting](https://dI.acm.org/doi/pdf/10.1145/3447548.3467184)|KDD|2021|[Code link](https://github.com/maryam-tabar/PLAN)| - | - |
| [Topological Attention for Time Series Forecasting](https://arxiv.org/pdf/2107.09031v1.pdf) | NeurIPS | 2021 | [Code link](https://github.com/ElementAl/N-BEATS) | [M4 competition dataset](https://github.com/Mcompetitions/M4-methods) | - |
| [MixSeq: Connecting Macroscopic Time Series Forecasting with Microscopic Time Series Data](https://arxiv.org/pdf/2110.14354v1.pdf) | NeurIPS | 2021 | - | [Rossmann](https://www.kaggle.com/c/rossmann-store-sales), Wiki and [M5](https://www.kaggle.com/c/m5-forecasting-accuracy) | - |
| [Test-time Collective Prediction](https://arxiv.org/pdf/2106.12012v1.pdf) | NeurIPS | 2021 | - | Boston...... | - |
| [Bubblewrap: Online tiling and real-time flow prediction on neural manifolds](https://arxiv.org/pdf/2108.13941v1.pdf) | NeurIPS | 2021 | [Code link](https://github.com/pearsonlab/Bubblewrap) | - | - |
| [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/pdf/2106.13008v2.pdf) | NeurIPS | 2021 | [Code link](https://github.com/thuml/Autoformer) | ETT, Electricity, Exchange, and Traffic | - |
| [Learning to Learn the Future: Modeling Concept Drifts in Time Series Prediction](https://dl.acm.org/doi/pdf/10.1145/3459637.3482271) |  CIKM | 2021 | - | Climate Dataset, Stock Dataset and Synthetic Dataset | - |
| [AdaRNN: Adaptive Learning and Forecasting of Time Series](https://arxiv.org/pdf/2108.04443.pdf) |  CIKM | 2021 | - | UCI activity, Air quality, Electric power and Stock price | - |
| [Actionable Insights in Urban Multivariate Time-series](https://dl.acm.org/doi/pdf/10.1145/3459637.3482410) |  CIKM | 2021 | - | Gaussian, Insect, Wikipedia and so on ...... | - |
| [Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting](https://arxiv.org/pdf/2103.16349.pdf) |  CIKM | 2021 | - | [ETT](https://github.com/zhouhaoyi/), [Electricity](https://github.com/laiguokun/multivariate-time-series-data)  | - |
| [AGCNT: Adaptive Graph Convolutional Network for Transformer-based Long Sequence Time-Series Forecasting](https://dl.acm.org/doi/pdf/10.1145/3459637.3482054) |  CIKM | 2021 | - | ETT | - |
| [PIETS: Parallelised Irregularity Encoders for Forecasting with Heterogeneous Time-Series](https://arxiv.org/pdf/2110.00071.pdf) |  ICDM | 2021 | - | Covid-19 | - |
| [Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting](https://arxiv.org/pdf/2109.01876.pdf) |  ICDM | 2021 | [Code link](https://github.com/sheoyon-jhin/ANCDE) | Character Trajectories, PhysioNet Sepsis and Google Stock. | - |
| [SSDNet: State Space Decomposition Neural Network for Time Series Forecasting](https://arxiv.org/pdf/2112.10251.pdf) |  ICDM | 2021 | - | Electricity,Exchange, Solar ...... | - |
| [Two Birds with One Stone: Series Saliency for Accurate and Interpretable Multivariate Time Series Forecasting](https://www.ijcai.org/proceedings/2021/0397.pdf) |  IJCAI | 2021 | - | electricity, Air-quality, Industry data | - |
| [TE-ESN: Time Encoding Echo State Network for Prediction Based on Irregularly Sampled Time Series Data](https://arxiv.org/pdf/2105.00412.pdf) |  IJCAI | 2021 | - | MG system, SILSO, USHCN, COVID-19 | - |
| [DeepFEC: Energy Consumption Prediction under Real-World Driving Conditions for Smart Cities](https://dl.acm.org/doi/pdf/10.1145/3442381.3449983) | WWW | 2021| [Code link](https://github.com/ElmiSay/DeepFEC) |[SPMD](https://catalog.data.gov/dataset/safety-pilot-model-deployment-data), [VED](https://github.com/gsoh/VED) | - |
| [HINTS: Citation Time Series Prediction for New Publications viaDynamic Heterogeneous Information Network Embedding](https://dl.acm.org/doi/pdf/10.1145/3442381.3450107) | WWW | 2021| - | [the AMiner Computer Science dataset](https://aminer.org/citation) and [the American Physical Society (APS) Physics dataset](https://journals.aps.org/datasets) | - |
| [Variable Interval Time Sequence Modeling for Career Trajectory Prediction: Deep Collaborative Perspective](https://dl.acm.org/doi/pdf/10.1145/3442381.3449959) | WWW | 2021| - | traffic data from 1988.1 to 2018.11 | - |
| [REST: Reciprocal Framework for Spatiotemporal coupled predictions](https://dl.acm.org/doi/pdf/10.1145/3442381.3449928) | WWW | 2021| - |  a traffic dataset released by Li et al. and [a web dataset](https://dumps.wikimedia.org) | - |
| [AutoSTG: Neural Architecture Search for Predictions of Spatio-Temporal Graph](https://dl.acm.org/doi/pdf/10.1145/3442381.3449816) | WWW | 2021| [Code link](https://github.com/panzheyi/AutoSTG) | PEMS-BAY and METR-LA | - |
| [Fine-grained Urban Flow Prediction](https://dl.acm.org/doi/pdf/10.1145/3442381.3449792) | WWW | 2021| - | TaxiBJ+, HappyValley | - |
| [Probabilistic Time Series Forecasting with Shape and Temporal Diversity](https://proceedings.neurips.cc/paper/2020/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html) | NeurIPS | 2020| [Code link](https://github.com/vincent-leguen/STRIPE) | - | - |
| [Benchmarking Deep Learning Interpretability in Time Series Predictions](https://proceedings.neurips.cc/paper/2020/hash/47a3893cc405396a5c30d91320572d6d-Abstract.html) | NeurIPS | 2020| [Code link](https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark) | - | - |
| [Adversarial Sparse Transformer for Time Series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html) | NeurIPS | 2020| - | - | - |
| [Deep Rao-Blackwellised Particle Filters for Time Series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/afb0b97df87090596ae7c503f60bb23f-Abstract.html) | NeurIPS | 2020| - | - | - |
| [Gamma-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction](https://proceedings.neurips.cc/paper/2020/hash/12ffb0968f2f56e51a59a6beb37b2859-Abstract.html) | NeurIPS | 2020| - | - | - |
| [EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning](https://proceedings.neurips.cc/paper/2020/hash/e4d8163c7a068b65a64c89bd745ec360-Abstract.html) | NeurIPS | 2020| - | - | - |
| [Multi-agent Trajectory Prediction with Fuzzy Query Attention](https://proceedings.neurips.cc/paper/2020/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html) | NeurIPS | 2020| [Code link](https://github.com/nitinkamra1992/FQA) | - | - |
| [Set Functions for Time Series](http://proceedings.mlr.press/v119/horn20a.html) | ICML | 2020| [Code link](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series) | - | - |
| [Learning from Irregularly-Sampled Time Series: A Missing Data Perspective](http://proceedings.mlr.press/v119/li20k.html) | ICML | 2020| [Code link](https://github.com/steveli/partial-encoder-decoder) | - | - |
| [Unsupervised Transfer Learning for Spatiotemporal Predictive Networks](http://proceedings.mlr.press/v119/yao20a.html) | ICML | 2020| [Code link](https://github.com/thuml/transferable-memory) | - | - |
| [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://dl.acm.org/doi/10.1145/3394486.3403118) | KDD | 2020| [Code link](https://github.com/nnzhan/MTGNN) | - | - |
| [Deep State-Space Generative Model For Correlated Time-to-Event Predictions](https://dl.acm.org/doi/10.1145/3394486.3403206) | KDD | 2020| [Code link](https://github.com/Google-Health/records-research/state-space-model) | - | - |
| [Attention based multi-modal new product sales time-series forecasting](https://dl.acm.org/doi/10.1145/3394486.3403362) | KDD | 2020| - | - | - |
| [BusTr: predicting bus travel times from real-time traffic](https://dl.acm.org/doi/10.1145/3394486.3403376) | KDD | 2020| - | - | - |
| [CompactETA: A Fast Inference System for Travel Time Prediction](https://dl.acm.org/doi/10.1145/3394486.3403386) |KDD| 2020| - | - | - |
| [DATSING: Data Augmented Time Series Forecasting with Adversarial Domain Adaptation]() | CIKM | 2020| [Code link]() | - | - |
| [Dual Sequential Network for Temporal Sets Prediction](https://dl.acm.org/doi/pdf/10.1145/3397271.3401124) | SIGIR | 2020| [Code link]() | - | - |



## Time Series Classification 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [OMNI-SCALE CNNS: A SIMPLE AND EFFECTIVE KERNEL SIZE CONFIGURATION FOR TIME SERIES CLASSIFICATION](https://openreview.net/pdf?id=PDYs7Z2XFGv) | ICLR |2022 |  [Code link](https://github.com/Wensi-Tang/OS-CNN) | - |- |
| [Correlative Channel-Aware Fusion for Multi-View Time Series Classification](https://arxiv.org/abs/1911.11561) | AAAI | 2021 | - | EV-Action, NTU RGB+D, UCI Daily and Sports Activities | The global-local temporal encoders are developed to extract robust temporal representations for each view, and a learnable fusion mechanism is proposed to boost the multi-view label information. |
| [Learnable Dynamic Temporal Pooling for Time Series Classification](https://arxiv.org/abs/2104.02577) | AAAI | 2021 | - | UCR/UEA |proposes a dynamic temporal pooling + a learning framework to simultaneously optimize the network parameters of a CNN classifier and the prototypical hidden series that encodes the latent semantic of the segments. |
| [ShapeNet: A Shapelet-Neural Network Approach for Multivariate Time Series Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17018) | AAAI | 2021 | [Code link](http://alturl.com/d26bo) | UEA MTS datasets | We propose Mdc-CNN to learn time series subsequences of various lengths into unified space and propose a cluster-wise triplet loss to train the network in an unsupervised fashion. We adopt MST to obtain the MST representation of time series. |
| [Joint-Label Learning by Dual Augmentation for Time Series Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17071) | AAAI | 2021 | [Code](https://github.com/fchollet/keras) | [UCR](https://www.cs.ucr.edu/˜eamonn/) | a novel time-series data augmentation method  |
| [Explainable Multivariate Time Series Classification: A Deep Neural Network Which Learns To Attend To Important Variables As Well As Time Intervals](https://arxiv.org/pdf/2011.11631.pdf) | WSDM | 2021 | - | - |- |
| [Voice2Series: Reprogramming Acoustic Models for Time Series Classification](http://proceedings.mlr.press/v139/yang21j.html) | ICML |2021 |  [Code link](https://github.com/huckiyang/Voice2Series-Reprogramming) | - |- |
| [Learning Saliency Maps to Explain Deep Time Series Classifiers](https://thartvigsen.github.io/papers/cikm21.pdf) |  CIKM | 2021 | [Code link]() | - | - |
| [Gaussian Process Model Learning for Time Series Classification](https://dl.acm.org/doi/pdf/10.1145/3468791.3468839) |  ICDM | 2021 | [Code link]() | - | - |
| [Contrast Profile: A Novel Time Series Primitive that Allows Classification in Real World Settings](https://link.springer.com/content/pdf/10.1007/s10618-020-00695-8.pdf) |  ICDM | 2021 | [Code link]() | - | - |
| [Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting](https://arxiv.org/pdf/2109.01876.pdf) |  ICDM | 2021 | [Code link]() | - | - |
| [Imbalanced Time Series Classification for Flight Data Analyzing with Nonlinear Granger Causality Learning](https://dl.acm.org/doi/pdf/10.1145/3340531.3412710) | CIKM | 2020| [Code link]() | - | - |
| [Visualet: Visualizing Shapelets for Time Series Classification](https://dl.acm.org/doi/pdf/10.1145/3340531.3417414) | CIKM | 2020| [Code link]() | - | - |
| [Learning Discriminative Virtual Sequences for Time Series Classification](https://dl.acm.org/doi/pdf/10.1145/3340531.3412099) | CIKM | 2020| [Code link]() | - | - |
| [Fast and Accurate Time Series Classification Through Supervised Interval Search](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338332) | CIKM | 2020| [Code link]() | - | - |






## Anomaly Detection

|             Dataset             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :------------------------: | ----------------------- | ------------------------- |------ |
| [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/abs/2109.05257) | AAAI | 2022 | - | Secure water treatment (SWaT), ...... | applying PA can severely overestimate a TAD model’s capability.|
| [DeepGPD: A Deep Learning Approach for Modeling Geospatio-Temporal Extreme Events](https://aaai-2022.virtualchair.net/poster_aaai10861) | AAAI | 2022 | [Code link](https://github.com/TylerPWilson/deepGPD) | [the Global Historical Climatology Network (GHCN)](https://www.ncdc.noaa.gov/ghcn-daily-description) |proposed a novel deep learning architecture (DeepGPD) capable of learning the parameters of the generalized Pareto distribution while satisfying the conditions placed on those parameters.|
| [Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series](https://arxiv.org/abs/2202.07857) | ICLR |2022 |  - | PMU-B, PMU-C, SWaT, METR-LA | - |
| [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642) | ICLR |2022 |  [Code link]() | - | - |
| [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://www.aaai.org/AAAI21Papers/AAAI-5076.DengA.pdf) | AAAI | 2021 | - | - | - | - |
| [Time Series Anomaly Detection with Multiresolution Ensemble Decoding](https://www.aaai.org/AAAI21Papers/AAAI-5192.ShenL.pdf) | AAAI | 2021 | - | - | - |- |
| [Outlier Impact Characterization for Time Series Data](https://par.nsf.gov/servlets/purl/10272499) | AAAI | 2021 | [benchmark](https://github.com/numenta/NAB) | [Webscope](http://labs.yahoo.com/Academic-Relations), [Physionet](https://physionet.org/content/chfdb/1.0.0/) | - |
| [F-FADE: Frequency Factorization for Anomaly Detection in Edge Streams](https://arxiv.org/pdf/2011.04723.pdf) | WSDM | 2021 | [Code link](http://snap.stanford.edu/f-fade/) | - |- |
| [FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection](https://dl.acm.org/doi/pdf/10.1145/3437963.3441823) | WSDM | 2021 | [Code link](https://github.com/DawnsonLi/EVT) | - |- |
| [Event Outlier Detection in Continuous Time](https://proceedings.mlr.press/v139/liu21g.html) | ICML |2021 |  [Code link](https://github.com/siqil/CPPOD) | - |- |
| [Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding](https://dl.acm.org/doi/10.1145/3447548.3467075) |  KDD |2021 | [Code link](https:/github.com/zhhlee/lnterFusion) | - |- |
| [Practical Approach to Asynchronous Multi-variate Time Series Anomaly Detection and Localization](https://dl.acm.org/doi/10.1145/3447548.3467174) | KDD |2021 |  [Code link](https://github.com/eBay/RANSyncoders) | - |- |
| [Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering](https://dl.acm.org/doi/10.1145/3447548.3467137) |KDD | 2021 |  [Code link](https://github.com/NSIBF/NSIBF) | - |- |
| [Multi-Scale One-Class Recurrent Neural Networks for Discrete Event Sequence Anomaly Detection](https://dl.acm.org/doi/10.1145/3447548.3467125) | KDD |2021 |  [Code link](https://github.com/wzwtrevor/Multi-Scale-One-Class-Recurrent-Neural-Networks) | - |- |
| [Online false discovery rate control for anomaly detection in time series](https://assets.amazon.science/7f/93/7e61ec0143ce844f71d507e7185e/on-line-false-discovery-rate-control-for-anomaly-detection-in-time-series.pdf) | NeurIPS | 2021 | - | - | - |
| [Detecting Anomalous Event Sequences with Temporal Point Processes](https://arxiv.org/pdf/2106.04465.pdf) | NeurIPS | 2021 | - | - | - |
| [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/pdf/2106.00666.pdf) | NeurIPS | 2021 | [Code link](https://github.com/hustvl/YoLos) | - | - |
| [Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers](https://arxiv.org/pdf/2108.11996.pdf)| NeurIPS | 2021 | - | - | - |
| [Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/pdf/2107.03502.pdf)| NeurIPS | 2021 | [Code link](https:/github.com/ermongroup/CsDl) | - | - |
| [SDFVAE: Static and Dynamic Factorized VAE for Anomaly Detection of Multivariate CDN KPIs](https://dl.acm.org/doi/pdf/10.1145/3442381.3450013) | WWW | 2021| [Code link]() | - | - |
| [Time-series Change Point Detection with Self-Supervised Contrastive Predictive Coding](https://arxiv.org/pdf/2011.14097.pdf) | WWW | 2021| [Code link]() | - | - |
| [NTAM: Neighborhood-Temporal Attention Model for Disk Failure Prediction in Cloud Platforms](https://dl.acm.org/doi/pdf/10.1145/3442381.3449867) | WWW | 2021| [Code link]() | - | - |
| [One Detector to Rule Them All: Towards a General Deepfake Attack Detection Framework](https://arxiv.org/pdf/2105.00187.pdf) | WWW | 2021| [Code link]() | - | - |
| [Improving Irregularly Sampled Time Series Learning with Time-Aware Dual-Attention Memory-Augmented Networks](https://dl.acm.org/doi/pdf/10.1145/3459637.3482079) |  CIKM | 2021 | [Code link]() | - | - |
| [BiCMTS: Bidirectional Coupled Multivariate Learning of Irregular Time Series with Missing Values](https://dl.acm.org/doi/pdf/10.1145/3459637.3482064) |  CIKM | 2021 | [Code link]() | - | - |
| [Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network](https://proceedings.neurips.cc/paper/2020/hash/97e401a02082021fd24957f852e0e475-Abstract.html) | NeurIPS | 2020| - | - | - |
| [USAD : UnSupervised Anomaly Detection on multivariate time series](https://dl.acm.org/doi/10.1145/3394486.3403392) | KDD | 2020| - | - | - |
| [Application Performance Anomaly Detection with LSTM on Temporal Irregularities in Logs](https://hal.archives-ouvertes.fr/hal-03117074/document) | CIKM | 2020| [Code link]() | - | - |
| [Multivariate Time-series Anomaly Detection via Graph Attention Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338317) | ICDM | 2020| [Code link]() | - | - |
| [MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338376) | ICDM | 2020| [Code link]() | - | - |

## Time series Clustering 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Clustering Interval-Censored Time-Series for Disease Phenotyping](https://arxiv.org/abs/2102.07005v4) | AAAI | 2022 | - | - |
| [Corsets for Time Series Clustering](https://arxiv.org/pdf/2110.15263.pdf) | NeurIPS| 2021 | - | - |-|
| [Temporal Phenotyping using Deep Predictive Clustering of Disease Progression](http://proceedings.mlr.press/v119/lee20h.html) | ICML | 2020| [Code link](https://github.com/chl8856/AC_TPC) | - | - |

## Time series Segmentation 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [ClaSP-Time Series Segmentation](https://dl.acm.org/doi/pdf/10.1145/3459637.3482240)| CIKM | 2021 | - | - | - |
| [Multi-series Time-aware Sequence Partitioning for Disease Progression Modeling](https://www.ijcai.org/proceedings/2021/0493.pdf)| IJCAI | 2021 | - | sEMG | - |


## Others

|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Conditional Loss and Deep Euler Scheme for Time Series Generation](https://arxiv.org/abs/2102.05313v5) | AAAI | 2022 | - | - | - |
| [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) | AAAI | 2022 | [Code link](https://github.com/yuezhihan/ts2vec) | [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/),[30 UEA datasets](http://www.timeseriesclassification.com/), [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset), [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70), [KPI dataset](http://test-10056879.file.myqcloud.com/10056879/test/20180524_78431960010324/KPI%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%86%B3%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86.zip)  | performs contrastive learning in a hierarchical way over augmented context views|
| [Time Masking for Temporal Language Models](https://dl.acm.org/doi/abs/10.1145/3488560.3498529) | WSDM | 2022 | [Code link](https://github.com/guyrosin/tempobert) | - |- |
| [Long Short-Term Temporal Meta-learning in Online Recommendation](https://dl.acm.org/doi/10.1145/3488560.3498371) | WSDM | 2022 | - | - |- |
| [Structure Meets Sequences: Predicting Network of Co-evolving Sequences](https://dl.acm.org/doi/10.1145/3488560.3498411) | WSDM | 2022 | [Code link](https://github.com/SoftWiser-group/SeeS) | - |- |
| [EvoKG: Jointly Modeling Event Time and Network Structure for Reasoning over Temporal Knowledge Graphs](https://dl.acm.org/doi/10.1145/3488560.3498451) | WSDM | 2022 | [Code link](https://namyongpark.github.io/evokg) | - | - |
| [FILLING THE G AP S: MULTIVARIATE TIME SERIES IMPUTATION BY GRAPH NEURAL NETWORKS](https://openreview.net/pdf?id=kOu3-S3wJ7) |  ICLR |2022 | - | Air quality, Traffic, and Smart Grids | - |
| [PSA-GAN: PROGRESSIVE SELF ATTENTION GANS FOR SYNTHETIC TIME SERIES](https://openreview.net/pdf?id=Ix_mh42xq5w) |  ICLR |2022 | [Code link](https://github.com/mbohlkeschneider/psa-gan), [Code on glueonts](https://github.com/awslabs/gluon-ts.) | Electricty, M4, Solar energy, Traffic | - |
| [Generative Semi-Supervised Learning for Multivariate Time Series Imputation](https://www.aaai.org/AAAI21Papers/AAAI-7391.MiaoX.pdf) | AAAI | 2021 | - | - | - |
| [Temporal Cross-Effects in Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3437963.3441802) | WSDM | 2021 | - | - |- |
| [Learning Dynamic Embeddings for Temporal Knowledge Graphs](https://dl.acm.org/doi/pdf/10.1145/3437963.3441741) | WSDM | 2021 | - | - |- |
| [Temporal Meta-path Guided Explainable Recommendation](https://arxiv.org/pdf/2101.01433.pdf) | WSDM | 2021 | - | - |- |
| [Generative Adversarial Networks for Markovian Temporal Dynamics: Stochastic Continuous Data Generation](https://proceedings.mlr.press/v139/park21d.html) | ICML | 2021 | [Code link]() | - |- |
| [Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning](https://dl.acm.org/doi/10.1145/3447548.3467422) |  KDD | 2021 |[Code link](https://github.com/marlin-codes/HTGN-KDD21) | - |- |
| [Time-series Generation by Contrastive Imitation](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=26999) | NeurIPS | 2021 | - | - |- |
|[Adjusting for Autocorrelated Errors in Neural Networks for Time Series](https://arxiv.org/pdf/2101.12578.pdf)| NeurIPS | 2021 | [Code link](https://github.com/Daikon-Sun/AdjustAutocorrelation) | - |- |
| [Spikelet: An Adaptive Symbolic Approximation for Finding Higher-Level Structure in Time Series](https://ieeexplore.ieee.org/document/9679141) |  ICDM | 2021 | - | - | - |
| [STING: Self-attention based Time-series Imputation Networks using GAN](https://ieeexplore.ieee.org/document/9679183) |  ICDM | 2021 | [Code link]() | - | - |
| [SMATE: Semi-Supervised Spatio-Temporal Representation Learning on Multivariate Time Series](https://arxiv.org/pdf/2110.00578.pdf) |  ICDM | 2021 | [Code link]() | - | - |
| [TCube: Domain-Agnostic Neural Time-series Narration](https://arxiv.org/pdf/2110.05633.pdf) |  ICDM | 2021 | [Code link]() | - | - |
| [Towards Interpretability and Personalization: A Predictive Framework for Clinical Time-series Analysis](https://ieeexplore.ieee.org/document/9679181) |  ICDM | 2021 | [Code link]() | - | - |
| [Continual Learning for Multivariate Time Series Tasks with Variable Input Dimensions](https://arxiv.org/pdf/2203.06852.pdf) |  ICDM | 2021 | [Code link]() | - | - |
| [CASPITA: Mining Statistically Significant Paths in Time Series Data from an Unknown Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679098) |  ICDM | 2021 | [Code link]() | - | - |
| [Multi-way Time Series Join on Multi-length Patterns](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679018) |  ICDM | 2021 | [Code link]() | - | - |
| [Temporal Event Profiling based on Multivariate Time Series Analysis over Long-term Document Archives](https://link.springer.com/content/pdf/10.1186/s13059-015-0639-8.pdf) |  SIGIR | 2021 | [Code link]() | - | - |
| [Time-Aware Multi-Scale RNNs for Time Series Modeling](https://www.ijcai.org/proceedings/2021/0315.pdf) |  ICDM | 2021 | [Code link]() | - | - |
| [Deep reconstruction of strange attractors from time series](https://proceedings.neurips.cc/paper/2020/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html) | NeurIPS | 2020| [Code link](https://github.com/williamgilpin/fnn) | - | - |
| [High-recall causal discovery for autocorrelated time series with latent confounders](https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html) | NeurIPS | 2020| [Code link](https://github.com/jakobrunge/tigramite) | - | - |
| [Learning Long-Term Dependencies in Irregularly-Sampled Time Series](https://arxiv.org/abs/2006.04418) | NeurIPS | 2020| [Code link](https://github.com/mlech26l/ode-lstms) | - | - |
| [ARMA Nets: Expanding Receptive Field for Dense Prediction](https://proceedings.neurips.cc/paper/2020/hash/cd10c7f376188a4a2ca3e8fea2c03aeb-Abstract.html) | NeurIPS | 2020| [Code link](https://github.com/umd-huang-lab/ARMA-Networks) | - | - |
| [Learnable Group Transform For Time-Series](http://proceedings.mlr.press/v119/cosentino20a.html) | ICML | 2020| [Code link](https://github.com/Koldh/LearnableGroupTransform-TimeSeries) | - | - |
| [Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns](https://dl.acm.org/doi/10.1145/3394486.3403271) | KDD | 2020| - | - | - |
| [Matrix Profile XXI: A Geometric Approach to Time Series Chains Improves Robustness](https://dl.acm.org/doi/10.1145/3394486.3403164) | KDD | 2020| [Code link](https://sites.google.com/site/timeserieschains) | - | - |
| [Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data](https://dl.acm.org/doi/10.1145/3394486.3403228) | KDD | 2020| [Code link](https://github.com/floft/codats) | - | - |
| [Personalized Imputation on Wearable-Sensory Time Series via Knowledge Transfer](https://dl.acm.org/doi/pdf/10.1145/3340531.3411879) | CIKM | 2020| [Code link]() | - | - |
| [Hybrid Sequential Recommender via Time-aware Attentive Memory Network](https://dl.acm.org/doi/pdf/10.1145/3340531.3411869) | CIKM | 2020| [Code link]() | - | - |
| [Order-Preserving Metric Learning for Mining Multivariate Time Series](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338310) | ICDM | 2020| [Code link]() | - | - |
| [Fast Automatic Feature Selection for Multi-period Sliding Window Aggregate in Time Series](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338337) | ICDM | 2020| [Code link]() | - | - |
| [Matrix Profile XXII: Exact Discovery of Time Series Motifs Under DTW](https://ieeexplore.ieee.org/document/9338266) | ICDM | 2020| [Code link]() | - | - |
| [Inductive Granger Causal Modeling for Multivariate Time Series](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338381) | ICDM | 2020 | [Code link]() | - | - |
| [Mining Recurring Patterns in Real-Valued Time Series using the Radius Profile](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338407) | ICDM | 2020 | [Code link]() | - | - |
| [Learning Periods from Incomplete Multivariate Time Series](cs.albany.edu/~petko/lab/papers/zgzb2020icdm.pdf) | ICDM | 2020 | [Code link]() | - | - |
| [FilCorr: Filtered and Lagged Correlation on Streaming Time Series](https://ieeexplore.ieee.org/document/9338257) | ICDM | 2020 | [Code link]() | - | - |

