
import DataManager

if __name__=='__main__':
    dropFH = open('drop.txt', 'r')
    rawLines = dropFH.readlines()
    dropFH.close()
    dropList = [s.rstrip('\n').rstrip('\t') for s in rawLines]
    
    df = DataManager.DM('wave_5_elsa_data_v4.tab', 0.8, dropList) \
        .getFilteredDF()
    df.to_csv('hope.csv')
    
    