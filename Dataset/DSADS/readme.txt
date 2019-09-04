This directory contains the cross-position activity recognition datasets used in the following paper. Please consider citing this article if you want to use the datasets.

Jindong Wang, Yiqiang Chen, Lisha Hu, Xiaohui Peng, and Philip S. Yu. Stratified Transfer Learning for Cross-domain Activity Recognition. 2018 IEEE International Conference on Pervasive Computing and Communications (PerCom).

These datasets are secondly constructed based on three public datasets:
OPPORTUNITY (opp) [1], PAMAP2 (pamap2) [2], and UCI DSADS (dsads) [3].

------------------------------------------------------

Here are some useful information about this directory. Please feel free to contact jindongwang@outlook.com for more information.

1. This is NOT the raw data, since I have performed feature extraction and normalized the features into [-1,1]. The code for feature extraction can be found in here: https://github.com/jindongwang/activityrecognition/tree/master/code. Currently, there are 27 features for a single sensor. There are 81 features for a body part. More information can be found in above PerCom-18 paper.

2. There are 4 .mat files corresponding to each dataset: dsads.mat for UCI DSADS, opp_hl.mat and opp_ll.mat for OPPORTUNITY, and pamap.mat for PAMAP2. Note that opp_hl and opp_loco denotes 'high-level' and 'locomotion' activities, respectively.
(1) dsads.mat: 9120 * 408. Columns 1~405 are features, listed in the order of 'Torso', 'Right Arm', 'Left Arm', 'Right Leg', and 'Left Leg'. Each position contains 81 columns of features. Columns 406~408 are labels. Column 406 is the activity sequence indicating the executing of activities (usually not used in experiments). Column 407 is the activity label (1~19). Column 408 denotes the person (1~8).
(2) opp_hl.mat and opp_loco.mat: Same as dsads.mat. But they contain more body parts: 'Back', 'Right Upper Arm', 'Right Lower Arm', 'Left Upper Arm', 'Left Lower Arm', 'Right Shoe (Foot)', and 'Left Shoe (Foot)'. Of course we did not use the data of both shoes in our paper. Column 460 is the activity label (please refer to OPPORTUNITY dataset to see the meaning of those activities). Column 461 is the activity drill (also check the dataset information). Column 462 denotes the person (1~4).
(3) pamap.mat: 7312 * 245. Columns 1~243 are features, listed in the order of 'Wrist', 'Chest', and 'Ankle'. Column 244 is the activity label. Column 245 denotes the person (1~9).

2. There are another 3 datasets with the prefix 'cross_', containing only 4 common classes of each dataset. This is for experimenting the cross-dataset activity recognition (see our PerCom-18 paper). The 4 common classes are lying, standing, walking, and sitting.
(1) cross_dsads.mat: 1920*406. Columns 1~405 are features. Column 406 is labels.
(2) cross_opp.mat: 5022*460. Columns 1~459 are features. Column 460 is labels.
(3) cross_pamap.mat: 3063 * 244. Columns 1~243 are features. Column 244 is labels.

-------- Original references for the 3 datasets:

[1] R. Chavarriaga, H. Sagha, A. Calatroni, S. T. Digumarti, G. Troster, ¨
J. d. R. Millan, and D. Roggen, “The opportunity challenge: A bench- ´
mark database for on-body sensor-based activity recognition,” Pattern
Recognition Letters, vol. 34, no. 15, pp. 2033–2042, 2013.

[2] A. Reiss and D. Stricker, “Introducing a new benchmarked dataset
for activity monitoring,” in Wearable Computers (ISWC), 2012 16th
International Symposium on. IEEE, 2012, pp. 108–109.

[3] B. Barshan and M. C. Yuksek, “Recognizing daily and sports activities ¨
in two open source machine learning environments using body-worn
sensor units,” The Computer Journal, vol. 57, no. 11, pp. 1649–1667,
2014.