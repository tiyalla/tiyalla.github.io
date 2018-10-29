+++
date = "2018-10-28"
title = "Predictive Analytics in HealthCare"
math = "true"

+++

### Introduction
Over the years, the healthcare industry and its organizations have invested millions of dollars in building data warehouses and acquiring data analysts – for the sole purpose of understanding their data, and using this understanding to improve a patient’s life. The historical problem with having these data warehouses is that they are not actionable; they tell what is happening but not "why". 


#### What is Predictive Analytics
Predictive analytics is the practical result of Big Data and Business intelligence. It provides a “how” and “why” on data stored in data warehouses and other places. 
It does this by using current and historical data, statistical techniques from machine learning, modeling and data mining to analyze and make predictions about the future, or other unknown events. 

**How?** - This historical data is fed into a mathematical model that considers key patterns and trends. The model is then applied to current data to predict what happens next.

![](images/predictive.png)
<p align="center"> **Figure 1: How predictive analytics occurs.** </p>
##### Why Healthcare
In healthcare, the importance of being one step ahead of events is most clearly seen in the realms of intensive care, surgery, or emergency care, where a patient’s life might depend on a quick reaction time and a finely-tuned sense of when something is going wrong. Predictive analytics allows clinicians, financial experts, and administrative staff to receive alerts about potential events before they happen, and therefore make more informed choices about how to proceed with a decision. 

Instead of simply presenting information about past events to a user, predictive analytics estimate the likelihood of a future outcome based on patterns in the historical data. 

##### Current Uses of Predictive Analytics in Healthcare.

###### *Getting ahead of patient deterioration.*
Patients face a number of potential threats to their wellbeing while still at the hospital, some of which includes the development of sepsis, an intractable infection, or a sudden downturn due to their existing clinical conditions.

Data analytics can help providers react as quickly as possible to changes in a patient’s vitals, and may be able to identify an upcoming deterioration before symptoms clearly manifest themselves to the naked eye. 

A 2017 study explained that at the [University of Pennsylvania](https://www.newswise.com/articles/machine-learning-may-help-in-early-identification-of-severe-sepsis), a predictive analytics tool leveraging machine learning and electronic health record (EHR) data helped to identify patients on track for severe sepsis or septic shock 12 hours before the onset of the condition.

###### *Preventing appointment no shows.*
Unexpected gaps in the daily calendar can have financial repercussions for the organization while throwing off a clinician’s entire workflow. Using predictive analytics to identify patients likely to skip an appointment without advanced notice can improve provider satisfaction, cut down on revenue losses, and give organizations the opportunity to offer open slots to other patients, and send reminders to patients at risk of not showing up, thereby increasing speedy access to care.

According to a [study](https://healthitanalytics.com/news/predictive-analytics-ehr-data-identify-appointment-no-shows) from Duke university, a team found that predictive models using clinic-level data could capture an additional 4800 patient no-shows per year with higher accuracy than previous attempts to forecast patient patterns.

###### *Get ahead of no shows.*
In addition to helping organizations get ahead of no-shows, predictive analytics can give providers a heads up when the clinic is about to get busy. Care sites that operate without fixed schedules, such as emergency departments and urgent care centers, must vary their staffing levels to account for fluctuations in patient flow. Using analytics to predict patterns in utilization can help to ensure optimal staffing levels while reducing wait times and raising patient satisfaction.

###### *Risk scores for chronic diseases.*
Risk scores is a standardized metric for the likelihood that an individual will experience a particular outcome. In healthcare, these outcomes can include service utilization events, such as hospital admissions and emergency department visits, or the development of a certain clinical state, such as heart disease, diabetes, cancer, or sepsis.

Creating risk scores based on lab testing, biometric data, claims data, patient-generated health data, and the social determinants of health can give healthcare providers insight into which individuals might benefit from enhanced services or wellness activities.

<div align="center">
![](images/riskscores.png)

</div>
<p align="center"> **Figure 2: Example of a cardiovascular disease risk score**. </p>

##### Data Mining in Health Monitoring Systems
Recently, the research area of health monitoring systems has shifted from simple reasoning of wearable sensor readings (like calculating the sleep hours or the number of steps per day) to the higher level of data processing in order to give much more information that is valuable to the end users. 
Therefore, healthcare services have been focusing on deeper data mining tasks to have a deeper knowledge representation. Study has shown that in health monitoring systems there are three predominant tasks, these are: prediction, anomaly detection and decision making in diagnosis.

*Anomaly detection* is the task of identifying unusual patterns which do not conform to expected behavior of the data.

*Prediction*, as explained earlier identifies events which have not yet occurred. 

*Decision making in diagnosis* is one of the main tasks of clinical monitoring systems which is often based on retrieved knowledge using vital signs, and also other information such as electronic health records and metadata.

<div align="center">
![](images/tasks.png)

</div>
<p align="center"> **Figure 3: Predominant data mining tasks in wearable health monitoring devices with sensors in relation to three dimensions**. </p>

The first dimension involves the setting in which the monitoring occurs. Most monitoring applications which consider home settings or remote monitoring deal predominantly with prediction and anomaly detection whereas the applications in clinical settings are typically focused on diagnosis.

The second dimension shows the tasks with respect to the type of subjects used. For patients with known medical records, both diagnosis and specifically the possibility to raise alarms are key tasks. For health monitoring which typically include healthy individuals who want to ensure the maintenance of good health, prediction and anomaly detection are used in the 

The final dimension depicted in the Figure considers the three main data mining tasks in relation to how the data is processed. For all three tasks data has been addressed both in an online and offline manner, with more alarm related tasks being naturally used in the context of online and continuous monitoring.

##### Data Mining Approach
Extracting information from the low level sensor data and bridging them to the high level knowledge representation is a very important aspect of data analysis in health monitoring systems. 

Regardless of the data mining technique used, the most standard and widely used approach to mining information from wearable sensors is shown below:

<div align="center">
![](images/approach.png)

</div>
<p align="center"> **Figure 4: A generic architecture of the main data mining approach for wearable sensor data**. </p>

In the figure, the raw sensor data is typically used as a starting point of the data mining approach. Here, the sensor data is provided for both training data in order to learn the system, make a model of features, as well as testing data for real-world usage designed model and make the result. This data mining approach is suggested as a general flow for both supervised and unsupervised data mining solutions in order to provide any kind of data mining task as result. 

The main steps of the data mining approach consist (1) data preprocessing; (2) feature extraction and selection; and (3) modelling data learning the input features to perform the tasks such as detection, prediction, and decision making. 

Preprocessing - This step is used to filter unusual data and artifacts in sensor data. Unusual data and artifacts include occurrences of noise, motion artifacts and sensor errors etc. This step is very important.

Feature extraction - Generally, for mining massive and real world data sets, the abstraction of raw data in any data mining approach is a way to design and build a model in order to retrieve valuable information. The aim of feature extraction is to discover the main characteristics of a data set which are identically representatives of the original data.

Modeling - This step is used to make sense of the data. This is done using appropriate data processing techniques. Some of the most common algorithm used for modeling wearable sensor data are: Neural Netowrks, Decision Trees, Gaussian Mixture Models.

##### Challenges

###### Adoption
It’s not a secret that the more difficult a new technology is to use, the less likely end users are to adopt it—and predictive analytics solutions are notoriously difficult in meeting this challenge. This is because they typically live as standalone tools, which means users have to switch from their primary business application over to the predictive analytics solution in order to use it. Also, leveraging large data sets successfully requires a hospital system to be prepared to embrace new methodologies; this, however, may require a significant investment of time and capital and alignment of economic interests.

###### Expertise
Expertise is a challenge because predictive analytics solutions are typically designed for data scientists who have deep understanding of statistical modeling, R, and Python. This is inherently limiting. In fact, most end users can’t even begin to approach predictive analytics without first hiring a dedicated data scientist.

##### Other Uses of Predictive Analytics in other domains.
###### Retail
By incorporating retail analytics into predictive models, you can more readily foresee customers’ needs and encourage shoppers to come back for a personalized experience. Also, using predictive analytics grants you a path to both reduce expenses on inventory and ensure that the stock you’re buying converts into sales instead of sunk costs. Retailers who deploy analytics can focus their efforts to highlight areas of high demand, quickly pick up on emerging sales trends, and optimize delivery to ensure the right inventory goes to the correct store.

###### Detecting insurance fraud
Proving claims are fraudulent can in turn costs the companies more than the original cost of the claim itself, that's why the turn to predictive model to prove fraud. It also helps insurance companies understand and model the future behavior of their customers, enabling them to create tailored financial products that are relevant and appealing to them, while also being competitive in the marketplace.

##### References

1 - Banaee, Hadi, et al. “Data Mining for Wearable Sensors in Health Monitoring Systems: A Review of Recent Trends and Challenges.” MDPI, Multidisciplinary Digital Publishing Institute, 17 Dec. 2013, www.mdpi.com/1424-8220/13/12/17472/htm.

2 - Bharadwaj, Raghav. “Predictive Analytics in Healthcare - Current Applications and Trends.” TechEmergence, 10 Sept. 2018, www.techemergence.com/predictive-analytics-in-healthcare-current-applications-and-trends.

3 - HealthITAnalytics. “10 High-Value Use Cases for Predictive Analytics in Healthcare.” HealthITAnalytics, 4 Sept. 2018, www.healthitanalytics.com/news/10-high-value-use-cases-for-predictive-analytics-in-healthcare.

4 - “What Is Predictive Analytics? 3 Real-World Examples of Predictive Analytics in Business Intelligence.” Logi Analytics, 6 Aug. 2018, www.logianalytics.com/bi-trends/what-is-predictive-analytics/.

5 - “The 4 Common Challenges of Predictive Analytics & Possible Solutions.” Logi Analytics, 12 July 2018, www.logianalytics.com/embedded/the-4-common-challenges-of-predictive-analytics/.


