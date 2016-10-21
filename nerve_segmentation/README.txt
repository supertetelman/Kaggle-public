Description from:
https://www.kaggle.com/c/ultrasound-nerve-segmentation

Identify nerve structures in ultrasound images of the neck

Even the bravest patient cringes at the mention of a surgical procedure. Surgery inevitably brings discomfort, and oftentimes involves significant post-surgical pain. Currently, patient pain is frequently managed through the use of narcotics that bring a bevy of unwanted side effects.

This competition's sponsor is working to improve pain management through the use of indwelling catheters that block or mitigate pain at the source. Pain management catheters reduce dependence on narcotics and speed up patient recovery.

Accurately identifying nerve structures in ultrasound images is a critical step in effectively inserting a patient’s pain management catheter. In this competition, Kagglers are challenged to build a model that can identify nerve structures in a dataset of ultrasound images of the neck. Doing so would improve catheter placement and contribute to a more pain free future. 

Started: 2:00 pm, Thursday 19 May 2016 UTC 
Ends: 11:59 pm, Thursday 18 August 2016 UTC (91 total days) 

Data:
https://www.kaggle.com/c/ultrasound-nerve-segmentation/data

Notes:

*When doctors are using ultrasound to identify nerve structures they are not looking at single images. Rather they are looking at how structures within the image move between ultrasound readings (source: several youtube videos)
*The labelers of the competition only had still frames
*There is no order
*The test images do not have patient labels

Tasks:
c)Install opencv
p)Install Theano
o)install tensorflow
---
A lot of the work on this project relied on image/video transformations for insight, and CNN for analysis. There was a lot of very good collaboration on the Kaggle forums on both of these efforts and I relied heavily on a several of the analysis and CNN frameworks others had provided for my work.
