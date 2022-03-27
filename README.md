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
| [ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction](https://dl.acm.org/doi/10.1145/3488560.3498444) | WSDM | 2022 | [Code link](https://github.com/k51/STGSP) | - |- |
| [Time-Series Event Prediction with Evolutionary State Graph](https://arxiv.org/pdf/1905.05006.pdf) | WSDM | 2021 | [Code link](https://github.com/VachelHU/EvoNet) | - |- |
| [Long Horizon Forecasting With Temporal Point Processes](https://arxiv.org/pdf/2101.02815.pdf) | WSDM | 2021 | - | - |- |
| [Modeling Inter-station Relationships with Attentive Temporal Graph Convolutional Network for Air Quality Prediction](https://dl.acm.org/doi/pdf/10.1145/3437963.3441731) | WSDM | 2021 | - | - |- |
| [Predicting Crowd Flows via Pyramid Dilated Deeper Spatial-temporal Network](https://dl.acm.org/doi/pdf/10.1145/3437963.3441785) | WSDM | 2021 | - | - |- |
| [PYRAFORMER: LOW-COMPLEXITY PYRAMIDAL ATTENTION FOR LONG-RANGE TIME SERIES MODELING AND FORECASTING](https://openreview.net/pdf?id=0EXmFzUn5I) | ICLR | 2022 | [Code link](https://github.com/alipay/Pyraformer) | Electricity, Wind and App Flow | - |
| [DEPTS: DEEP EXPANSION LEARNING FOR PERIODIC TIME SERIES FORECASTING](https://openreview.net/pdf?id=AJAR-JgNw__) | ICLR | 2022 | [Code link](https://github.com/weifantt/DEPTS) | ELECTRICITY, TRAFFIC2, and M4(HOURLY) | - |
| [TAMP-S2GCNETS: COUPLING TIME-AWARE MULTIPERSISTENCE KNOWLEDGE REPRESENTATION WITH SPATIO-SUPRA GRAPH CONVOLUTIONAL NETWORKS FOR TIME-SERIES FORECASTING](https://openreview.net/pdf?id=wv6g8fWLX2q) | ICLR | 2022 | [Code link](https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0.) | PeMSD3, PeMSD4, PeMSD8 and COVID-19 | - |
| [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://arxiv.org/abs/2202.01575) | ICLR | 2022 | [Code link](https://github.com/salesforce/CoST) | [ETT](https://github.com/zhouhaoyi/ETDataset),[Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014),[Weather](https://www.ncei.noaa.gov/data/local-climatological-data/) |- |
| [REVERSIBLE INSTANCE NORMALIZATION FOR ACCURATE TIME-SERIES FORECASTING AGAINST DISTRIBUTION SHIFT](https://openreview.net/pdf?id=cGDAkQo1C0p) | ICLR | 2022 | - | [ETT](https://github.com/zhouhaoyi/ETDataset), [Electricity Consuming Load (ECL)](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |- |
| [TEMPORAL ALIGNMENT PREDICTION FOR SUPERVISED REPRESENTATION LEARNING AND FEW-SHOT SEQUENCE CLASSIFICATION](https://openreview.net/pdf?id=p3DKPQ7uaAi) | ICLR | 2022 | [Code link](https://github.com/BingSu12/TAP) | MSR Action3D, MSR Daily Activity3D, “Spoken Arabic Digits (SAD)” dataset, ChaLearn | - |
| [Voice2Series: Reprogramming Acoustic Models for Time Series Classification](http://proceedings.mlr.press/v139/yang21j.html) | ICML |2021 |  [Code link](https://github.com/huckiyang/Voice2Series-Reprogramming) | - |- |
| [Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting](https://proceedings.mlr.press/v139/chen21o.html) | ICML | 2021 | [Code link](https://github.com/Z-GCNETs/Z-GCNETs.git) | - |- |
| [Explaining Time Series Predictions with Dynamic Masks](https://proceedings.mlr.press/v139/crabbe21a.html) | ICML |2021 |  [Code link](https://github.com/JonathanCrabbe/Dynamask) | - |- |
| [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series](https://proceedings.mlr.press/v139/rangapuram21a.html) |  ICML |2021 | [Code link](https://github.com/awslabs/gluon-ts) | - |- |
| [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://proceedings.mlr.press/v139/rasul21a.html) | ICML |2021 |  - | - |- |
| [Conformal prediction interval for dynamic time-series](https://proceedings.mlr.press/v139/xu21h.html) | ICML |2021 |  [Code link](https://github.com/hamrel-cxu/EnbPI) | - |- |
| [RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting](https://proceedings.mlr.press/v139/pal21b.html) | ICML |2021 |  [Code link]() | - |- |
| [ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting](https://dl.acm.org/doi/10.1145/3447548.3467330) |  KDD |2021 | [Code link](https://github.com/JLDeng/ST-Norm) | - |- |
| [MiniRocket: A Fast (Almost) Deterministic Transform for Time Series Classification](https://dl.acm.org/doi/abs/10.1145/3447548.3467231) | KDD |2021 |  [Code link](https://github.com/angus924/minirocket) | - |- |
| [Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting](https://dl.acm.org/doi/10.1145/3447548.3467275)|KDD|2021|[Code link](https://github.com/liangzhehan/DMSTGCN)| - |- |
| [Forecasting Interaction Order on Temporal Graphs](https://dl.acm.org/doi/10.1145/3447548.3467341)|KDD|2021| [Code link](https://github.com/xiawenwen49/TAT-code)|-|-|
|[Quantifying Uncertainty in Deep Spatiotemporal Forecasting](https://dl.acm.org/doi/10.1145/3447548.3467325)| KDD|2021 | - | - | - |
|[Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting](https://dI.acm.org/doi/10.1145/3447548.3467430)|KDD|2021|[Code link](https://github.com/square-coder/STGODE)|-|-|
|[A PLAN for Tackling the Locust Crisis in East Africa: Harnessing Spatiotemporal Deep Models for Locust Movement Forecasting](https://dI.acm.org/doi/10.1145/3447548.3467184)|KDD|2021|[Code link](https://github.com/maryam-tabar/PLAN)| - | - |
| [Topological Attention for Time Series Forecasting](https://arxiv.org/pdf/2107.09031v1.pdf) | NeurIPS | 2021 | [Code link](https://github.com/ElementAl/N-BEATS) | - | - |
| [MixSeq: Connecting Macroscopic Time Series Forecasting with Microscopic Time Series Data](https://arxiv.org/pdf/2110.14354v1.pdf) | NeurIPS | 2021 | - | - | - |
| [Test-time Collective Prediction](https://arxiv.org/pdf/2106.12012v1.pdf) | NeurIPS | 2021 | - | - | - |
| [Bubblewrap: Online tiling and real-time flow prediction on neural manifolds](https://arxiv.org/pdf/2108.13941v1.pdf) | NeurIPS | 2021 | [Code link](https://github.com/pearsonlab/Bubblewrap) | - | - |
| [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/pdf/2106.13008v2.pdf) | NeurIPS | 2021 | [Code link](https://github.com/thuml/Autoformer) | - | - |
| [DeepFEC: Energy Consumption Prediction under Real-World Driving Conditions for Smart Cities]() | WWW | 2021| [Code link]() | - | - |
| [HINTS: Citation Time Series Prediction for New Publications viaDynamic Heterogeneous Information Network Embedding]() | WWW | 2021| [Code link]() | - | - |
| [Bid Prediction in Repeated Auctions with Learning]() | WWW | 2021| [Code link]() | - | - |
| [Outlier-Resilient Web Service QoS Prediction]() | WWW | 2021| [Code link]() | - | - |
| [Variable Interval Time Sequence Modeling for Career Trajectory Prediction: Deep Collaborative Perspective]() | WWW | 2021| [Code link]() | - | - |
| [REST: Reciprocal Framework for Spatiotemporal coupled predictions]() | WWW | 2021| [Code link]() | - | - |
| [AutoSTG: Neural Architecture Search for Predictions of Spatio-Temporal Graph]() | WWW | 2021| [Code link]() | - | - |
| [Fine-grained Urban Flow Prediction]() | WWW | 2021| [Code link]() | - | - |
| []() | WWW | 2021| [Code link]() | - | - |

## Time Series Classification 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Correlative Channel-Aware Fusion for Multi-View Time Series Classification](https://arxiv.org/abs/1911.11561) | AAAI | 2021 | - | EV-Action, NTU RGB+D, UCI Daily and Sports Activities | The global-local temporal encoders are developed to extract robust temporal representations for each view, and a learnable fusion mechanism is proposed to boost the multi-view label information. |
| [Learnable Dynamic Temporal Pooling for Time Series Classification](https://arxiv.org/abs/2104.02577) | AAAI | 2021 | - | UCR/UEA |proposes a dynamic temporal pooling + a learning framework to simultaneously optimize the network parameters of a CNN classifier and the prototypical hidden series that encodes the latent semantic of the segments. |
| [ShapeNet: A Shapelet-Neural Network Approach for Multivariate Time Series Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17018) | AAAI | 2021 | [Code link](http://alturl.com/d26bo) | UEA MTS datasets | We propose Mdc-CNN to learn time series subsequences of various lengths into unified space and propose a cluster-wise triplet loss to train the network in an unsupervised fashion. We adopt MST to obtain the MST representation of time series. |
| [Joint-Label Learning by Dual Augmentation for Time Series Classification](https://ojs.aaai.org/index.php/AAAI/article/view/17071) | AAAI | 2021 | [Code](https://github.com/fchollet/keras) | [UCR](https://www.cs.ucr.edu/˜eamonn/) | a novel time-series data augmentation method  |
| [Explainable Multivariate Time Series Classification: A Deep Neural Network Which Learns To Attend To Important Variables As Well As Time Intervals](https://arxiv.org/pdf/2011.11631.pdf) | WSDM | 2021 | - | - |- |
| [OMNI-SCALE CNNS: A SIMPLE AND EFFECTIVE KERNEL SIZE CONFIGURATION FOR TIME SERIES CLASSIFICATION](https://openreview.net/pdf?id=PDYs7Z2XFGv) | ICLR |2022 |  [Code link](https://github.com/Wensi-Tang/OS-CNN) | - |- |






## Anomaly Detection

|             Dataset             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :------------------------: | ----------------------- | ------------------------- |------ |
| [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/abs/2109.05257) | AAAI | 2022 | - | Secure water treatment (SWaT), ...... | applying PA can severely overestimate a TAD model’s capability.|
| [DeepGPD: A Deep Learning Approach for Modeling Geospatio-Temporal Extreme Events](https://aaai-2022.virtualchair.net/poster_aaai10861) | AAAI | 2022 | [Code link](https://github.com/TylerPWilson/deepGPD) | [the Global Historical Climatology Network (GHCN)](https://www.ncdc.noaa.gov/ghcn-daily-description) |proposed a novel deep learning architecture (DeepGPD) capable of learning the parameters of the generalized Pareto distribution while satisfying the conditions placed on those parameters.|
| [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://www.aaai.org/AAAI21Papers/AAAI-5076.DengA.pdf) | AAAI | 2021 | - | - | - | - |
| [Time Series Anomaly Detection with Multiresolution Ensemble Decoding](https://www.aaai.org/AAAI21Papers/AAAI-5192.ShenL.pdf) | AAAI | 2021 | - | - | - |- |
| [Outlier Impact Characterization for Time Series Data](https://par.nsf.gov/servlets/purl/10272499) | AAAI | 2021 | [benchmark](https://github.com/numenta/NAB) | [Webscope](http://labs.yahoo.com/Academic-Relations), [Physionet](https://physionet.org/content/chfdb/1.0.0/) | - |
| [F-FADE: Frequency Factorization for Anomaly Detection in Edge Streams](https://arxiv.org/pdf/2011.04723.pdf) | WSDM | 2021 | [Code link](http://snap.stanford.edu/f-fade/) | - |- |
| [FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection](https://dl.acm.org/doi/pdf/10.1145/3437963.3441823) | WSDM | 2021 | [Code link](https://github.com/DawnsonLi/EVT) | - |- |
| [Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series](https://arxiv.org/abs/2202.07857) | ICLR |2022 |  - | PMU-B, PMU-C, SWaT, METR-LA | - |
| [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642) | ICLR |2022 |  [Code link]() | - | - |
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
| [SDFVAE: Static and Dynamic Factorized VAE for Anomaly Detection of Multivariate CDN KPIs]() | WWW | 2021| [Code link]() | - | - |
| [Time-series Change Point Detection with Self-Supervised Contrastive Predictive Coding]() | WWW | 2021| [Code link]() | - | - |
| [NTAM: Neighborhood-Temporal Attention Model for Disk Failure Prediction in Cloud Platforms]() | WWW | 2021| [Code link]() | - | - |
| [One Detector to Rule Them All: Towards a General Deepfake Attack Detection Framework]() | WWW | 2021| [Code link]() | - | - |




## Time series Clustering 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Clustering Interval-Censored Time-Series for Disease Phenotyping](https://arxiv.org/abs/2102.07005v4) | AAAI | 2022 | - | - |
| [Corsets for Time Series Clustering](https://arxiv.org/pdf/2110.15263.pdf) | NeurIPS| 2021 | - | - |-|

## Time series Segmentation 
|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |


## Others

|             Paper             | Conference | Year | Code | Used Datasets |Key Contribution|
| :-------------------: | :----------: | :----------: | :------------------------: | ----------------------- |------ |
| [Conditional Loss and Deep Euler Scheme for Time Series Generation](https://arxiv.org/abs/2102.05313v5) | AAAI | 2022 | - | - | - |
| [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) | AAAI | 2022 | [Code link](https://github.com/yuezhihan/ts2vec) | [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/),[30 UEA datasets](http://www.timeseriesclassification.com/), [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset), [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014), [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70), [KPI dataset](http://test-10056879.file.myqcloud.com/10056879/test/20180524_78431960010324/KPI%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%86%B3%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86.zip)  | performs contrastive learning in a hierarchical way over augmented context views|
| [Generative Semi-Supervised Learning for Multivariate Time Series Imputation](https://www.aaai.org/AAAI21Papers/AAAI-7391.MiaoX.pdf) | AAAI | 2021 | - | - | - |
| [EvoKG: Jointly Modeling Event Time and Network Structure for Reasoning over Temporal Knowledge Graphs](https://dl.acm.org/doi/10.1145/3488560.3498451) | WSDM | 2022 | [Code link](https://namyongpark.github.io/evokg) | - | - |
| [Time Masking for Temporal Language Models](https://dl.acm.org/doi/abs/10.1145/3488560.3498529) | WSDM | 2022 | [Code link](https://github.com/guyrosin/tempobert) | - |- |
| [Long Short-Term Temporal Meta-learning in Online Recommendation](https://dl.acm.org/doi/10.1145/3488560.3498371) | WSDM | 2022 | - | - |- |
| [Structure Meets Sequences: Predicting Network of Co-evolving Sequences](https://dl.acm.org/doi/10.1145/3488560.3498411) | WSDM | 2022 | [Code link](https://github.com/SoftWiser-group/SeeS) | - |- |
| [Temporal Cross-Effects in Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3437963.3441802) | WSDM | 2021 | - | - |- |
| [Learning Dynamic Embeddings for Temporal Knowledge Graphs](https://dl.acm.org/doi/pdf/10.1145/3437963.3441741) | WSDM | 2021 | - | - |- |
| [Temporal Meta-path Guided Explainable Recommendation](https://arxiv.org/pdf/2101.01433.pdf) | WSDM | 2021 | - | - |- |
| [FILLING THE G AP S: MULTIVARIATE TIME SERIES IMPUTATION BY GRAPH NEURAL NETWORKS](https://openreview.net/pdf?id=kOu3-S3wJ7) |  ICLR |2022 | - | Air quality, Traffic, and Smart Grids | - |
| [PSA-GAN: PROGRESSIVE SELF ATTENTION GANS FOR SYNTHETIC TIME SERIES](https://openreview.net/pdf?id=Ix_mh42xq5w) |  ICLR |2022 | [Code link](https://github.com/mbohlkeschneider/psa-gan), [Code on glueonts](https://github.com/awslabs/gluon-ts.) | Electricty, M4, Solar energy, Traffic | - |
| [Generative Adversarial Networks for Markovian Temporal Dynamics: Stochastic Continuous Data Generation](https://proceedings.mlr.press/v139/park21d.html) | ICML | 2021 | [Code link]() | - |- |
| [Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning](https://dl.acm.org/doi/10.1145/3447548.3467422) |  KDD | 2021 |[Code link](https://github.com/marlin-codes/HTGN-KDD21) | - |- |
| [Time-series Generation by Contrastive Imitation](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=26999) | NeurIPS | 2021 | - | - |- |
|[Adjusting for Autocorrelated Errors in Neural Networks for Time Series](https://arxiv.org/pdf/2101.12578.pdf)| NeurIPS | 2021 | [Code link](https://github.com/Daikon-Sun/AdjustAutocorrelation) | - |- |

