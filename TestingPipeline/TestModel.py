import numpy as np
import pandas as pd

def unmatch(x):
    HFT = 'HFT'
    NON_HFT = 'NON HFT'
    MIX = 'MIX'
    if x == 1:
        return HFT
    elif x == 0:
        return NON_HFT
    elif x == -1:
        return MIX
    else:
        return "Dafuq?"

class TestModel():
    def __init__(self, traders, preds, threshold=None, foldername="Predictions/nonamepred.csv"):
        #self.model     = model
        self.preds     = preds
        self.traders   = traders
        self.threshold = threshold
        self.name      = foldername
    
    def CreateDataFrame(self):
        # Converts preds to pd dataframe:
        types = pd.DataFrame(data=self.preds, columns=['type'])
        
        # Adding the traders to this dataframe
        self.Ipreds = pd.concat([self.traders, types], axis = 1)
        
        return self.Ipreds
    
    def MajorityVote(self):
        if(self.Ipreds is None):
            print("You need to run the method `CreateDataFrame` before doing a majority vote!")
            return -1
        
        preds = [[]]

        for trader in self.Ipreds.Trader.unique():
            subset = self.Ipreds[self.Ipreds['Trader'] == trader]
            #print(self.PredsToString(subset).describe())

            pred = subset.mode()['type'][0]
            
            # Correcting the majority vote prediction:
            if(self.threshold is not None):
                if(pred == 1):
                    percentage = (subset['type'] == pred).mean()
                    # Checking if prediction for HFT is relevant:
                    if(percentage <= self.threshold):
                        print("Flipped {}".format(trader))
                        pred *= -1
                        
                elif(pred == -1):
                    percentage = (subset['type'] == pred).mean()
                    # Checking if prediction for MIX is relevant:
                    if(percentage <= self.threshold):
                        print("Flipped {}".format(trader))
                        pred *= -1


            preds.append([trader, pred])

        preds = preds[1:]
        
        ## Creating the dataframe
        self.Fpreds = pd.DataFrame(data=preds, columns=['Trader', 'type'])    

        return self.Fpreds
    
    def PredsToString(self, df):
        
        df['type'] = df['type'].apply(unmatch)
        return df
    
    def CreatePredCSV(self):
        # Creating the dataframe
        print("Creating the Dataframe of predictions:")
        print(self.CreateDataFrame().tail())
        print("\n")

        # Majority vote on each trader prediction
        print("Predicting value for each trader based on a majority vote:")
        print(self.MajorityVote().tail())
        print("\n")

        # Converting the predictions to string and saving the dataframe in csv format:
        print("Converting the predictions to string value:")
        self.Fpreds = self.PredsToString(self.Fpreds)
        print(self.Fpreds.tail())
        print("\n")        
        
        print("Saving them to {}".format(self.name))
        self.Fpreds.to_csv(self.name, index = False)
    