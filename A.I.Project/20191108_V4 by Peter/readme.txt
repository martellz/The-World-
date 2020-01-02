1. Dataset Source:
https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets

2. To bench the time needed, please open up the python scripts before and after each model, it should look like this:



import datetime
filepath = r"E:\My Work\timebench.txt"
with open(filepath,'a') as input_file:
    input_file.write("In Time of Naive Bayers: "+str(datetime.datetime.now())+'\n')
out_data = in_data



Change the second line to a target txt file of your choice. You will have to change the file paths for all scripts modules so you can review them in the same file.

P.S.1: When the project is opened, someone of the scripts will be run so there will be multiple lines of in/out-time before the actual training of the models. It is suggested to ignore / delete those lines before the training of the models.
P.S.2: To avoid sharing of processing power, train only one model at a time. Disable all the others linked modules to the second selector modules except the model that is to be trained by right clicking on the links and uncheck "Enable" to temporary shutdown the link between modules.



3. Manually calculate the time neede to time or use the Time Calculator included to calculate the time. Simply copy and paste the time listed in the other python scripts and paste them into the variables.