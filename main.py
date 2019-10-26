import os
import sys
import pandas
import dataProcessing as process

def main():
    try:
        path = sys.argv[1]
        print(path)
        if (os.path.isdir(path)):
            readwrite = process.Read_Write_Data(path)
            readwrite.read_in_files()
            data = readwrite.get_df()
            cleaner = process.Clean_Data(data)
            cleaner.clean_data()
            readwrite.df_to_csv("test.csv")
            cleaner.remove_stop_words()
            cleaner.lemmatization(True)
            print(readwrite.get_df())
            readwrite.df_to_csv("test-l.csv")
        else :
            print(path + " is not a valid directory")
    except Exception as e:
        print("Error running applicaton - have you include a filepath?: " + str(e))

main()
