
"""
Luis Sosa - Kmeans CLustering
July 20th, 2016

Using sklearn package in Python
In this code I am comparing sample 6 Test cohorts (which are not very good)
to 6 clusters I've created using K-Means clustering. The findings in this code
show that the test cohorts do not separate the data as well as the
Clusters I've created.

"""


import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import matplotlib.mlab as mlab
import numpy as np 
   

""" Here is where I initialize my lists for the events data """
test_cohort = [] 
event_name = []
user_id = []
whisper_id = []
extra_info = []
event_time = []


""" Here is where I initialize my lists to separate different cohorts """
A_time, A_eventname = [], []
B_time, B_eventname = [], []
C_time, C_eventname = [], []
D_time, D_eventname = [], []
E_time, E_eventname = [], []
F_time, F_eventname = [], []


""" This is where I read in my events data 
 and seperate the information by column name.
 I named my datafile "\data\"  """
with open('\data\.csv') as eventsdata:
    events = csv.reader(eventsdata)
    for row in events:
        test_cohort.append(row[0])
        event_name.append(row[1])
        user_id.append(row[2])
        whisper_id.append(row[3])
        extra_info.append(row[4])
        event_time.append(row[5])
        if (row[0] == 'A'):
            A_eventname.append(row[1])
            A_time.append(row[5])
        elif (row[0] == 'B'):
            B_eventname.append(row[1])
            B_time.append(row[5])
        elif (row[0] == 'C'):
            C_eventname.append(row[1])
            C_time.append(row[5])
        elif (row[0] == 'D'):
            D_eventname.append(row[1])
            D_time.append(row[5])
        elif (row[0] == 'E'):
            E_eventname.append(row[1])
            E_time.append(row[5])
        elif (row[0] == 'F'):
            F_eventname.append(row[1])
            F_time.append(row[5])
            

""" Converts the elements in my times lists for each cohort into variable type float
# I'm doing this as to calculate the mean for each cohort """
A_time = np.array(A_time).astype(np.float)
B_time = np.array(B_time).astype(np.float)
C_time = np.array(C_time).astype(np.float)
D_time = np.array(D_time).astype(np.float)
E_time = np.array(E_time).astype(np.float)
F_time = np.array(F_time).astype(np.float)


""" Here I calculate the mean times for each cohort"""
meanA = np.mean(A_time)  #1457269352130.17
meanB = np.mean(B_time)  #1457269449157.72
meanC = np.mean(C_time)  #1457266770873.82
meanD = np.mean(D_time)  #1457270391815.97
meanE = np.mean(E_time)  #1457265107036.84
meanF = np.mean(F_time)  #1457269579682.35
            

""" Record population average time (for reference)"""
population_avgtime = np.mean(np.array(event_time).astype(np.float))  # 1457268461397.03 


""" Record deviation (difference from cohort average time, to population average time) """
DeviationA = meanA-population_avgtime  #  890733.140869141
DeviationB = meanB-population_avgtime  #  987760.685058594
DeviationC = meanC-population_avgtime  # -1690523.21118164
DeviationD = meanD-population_avgtime  #  1930418.94262695 
DeviationE = meanE-population_avgtime  # -3354360.19482422
DeviationF = meanF-population_avgtime  #  1118285.32128906 
        

"""Population Standard Deviation of events_data """
SD_Events = np.std(np.array(event_time).astype(np.float))  #  179269564.00116479


""" Z-Scores for each cohort groups mean event_time """
Z_A = DeviationA/SD_Events  #  0.0049686802432528541
Z_B = DeviationB/SD_Events  #  0.0055099184881833921 
Z_C = DeviationC/SD_Events  # -0.0094300626034589854 
Z_D = DeviationD/SD_Events  #  0.010768246986812941
Z_E = DeviationE/SD_Events  # -0.018711264308972624 
Z_F = DeviationF/SD_Events  #  0.0062380099350372517 


"""Now Perform K-Means Clustering using K=6"""
event_name_num = []  
extra_info_num = []

"""Converted event_name category to numerical values for k means clustering """
for row in event_name:
    if (row == 'Whisper Created'):
        event_name_num.append(1)
    elif (row == 'Heart'):
        event_name_num.append(2)
    elif (row == 'Conversation Created'):
        event_name_num.append(3)


"""Converted extra_info category to numerical values for k means clustering """
for row in extra_info:
    if (row == 'reply'):
        extra_info_num.append(1)
    elif (row == 'undefined'):
        extra_info_num.append(2)
    elif (row == 'top-level'):
        extra_info_num.append(3)
        
""" Combine the columns for event time, event name, and event info into array """
kmeansdata = np.column_stack((event_time,event_name_num,extra_info_num))


""" Now create k means function and perform clustering algorithm """        
kmeans = KMeans(n_clusters=6, init='random', n_init=20)
kmeanresults = kmeans.fit(kmeansdata)


""" Displays the cluster centers created using algorithm """
kmeanresults.cluster_centers_
"""
array([[  1.45753626e+12,   1.98795202e+00,   1.86728204e+00],
       [  1.45702782e+12,   2.04369871e+00,   1.88876512e+00],
       [  1.45741757e+12,   2.00633334e+00,   1.87165104e+00],
       [  1.45713049e+12,   2.02381125e+00,   1.87494758e+00],
       [  1.45722140e+12,   1.99795305e+00,   1.86014147e+00],
       [  1.45731313e+12,   2.01688117e+00,   1.85887910e+00]])
"""
  
     

"""Place Average Event Time for each cluster I created into list """
cluster_avgtimes = [kmeanresults.cluster_centers_[0,0],kmeanresults.cluster_centers_[1,0],kmeanresults.cluster_centers_[2,0],kmeanresults.cluster_centers_[3,0],kmeanresults.cluster_centers_[4,0],kmeanresults.cluster_centers_[5,0]]


""" Sort this list from earliest time to latest """
cluster_avgtimes.sort()
        

""" Compute Z-Scores for Averages Times computed for each Cluster """
""" These Z-Scores are much more significant than Z-Scores for test_cohorts """
(cluster_avgtimes[0] - population_avgtime) / SD_Events ## -1.3423493544841858
(cluster_avgtimes[1] - population_avgtime) / SD_Events ## -0.76965232012383034
(cluster_avgtimes[2] - population_avgtime) / SD_Events ## -0.26249957416164332
(cluster_avgtimes[3] - population_avgtime) / SD_Events ##  0.24915161690728543
(cluster_avgtimes[4] - population_avgtime) / SD_Events ##  0.8317648038246217
(cluster_avgtimes[5] - population_avgtime) / SD_Events ##  1.4938312821445665




""" The blue lines in the center of the normal plot represents the average 
time for each test cohort. These six individual blue
lines are all centered around the population mean,
which suggests that the test cohorts do not show
any significant differences """
x = np.linspace(population_avgtime-3.5*SD_Events,population_avgtime+3.5*SD_Events,100)
plt.plot(x, mlab.normpdf(x,population_avgtime,SD_Events))
plt.title('Histogram of Test Cohort TImes')
plt.axvline(x=meanA)
plt.axvline(x=meanB)
plt.axvline(x=meanC)
plt.axvline(x=meanD)
plt.axvline(x=meanE)
plt.axvline(x=meanF)
plt.show()


""" This plot, which represents the average times
for the clusters I've created using the K-Means
clustering package in scikit, shows a much better
spread in regards to the different clusters. Notice
how the blue lines are all separated (not clumped 
together as in the previous plot) """
plt.plot(x, mlab.normpdf(x,population_avgtime,SD_Events))
plt.title('Histogram of My Cluster TImes')
plt.axvline(x=cluster_avgtimes[0])
plt.axvline(x=cluster_avgtimes[1])
plt.axvline(x=cluster_avgtimes[2])
plt.axvline(x=cluster_avgtimes[3])
plt.axvline(x=cluster_avgtimes[4])
plt.axvline(x=cluster_avgtimes[5])
plt.show()


        


    

        
    
