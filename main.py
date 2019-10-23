import os
import sys
import pandas
import preprocessing as pre

def main():
    try:
        path = sys.argv[1]
        print(path)
        if (os.path.isdir(path)):
            reader = pre.Read_Data(path)
            reader.read_in_files()
            data = reader.get_df()
            cleaner = pre.Clean_Data(data)
            cleaner.clean_data()
            cleaner.df_to_csv("test.csv")
            # cleaner.remove_stop_words()
        else :
            print(path + " is not a valid directory")
    except Exception as e:
        print("Error running applicaton - have you include a filepath?: " + str(e))

main()
