import os
import sys
import pandas
import dataProcessing as process

def main():
    output_csv = "test.csv"
    textRank = True
    try:
        path = sys.argv[1]
        print(path)
        try :
            output_csv = sys.argv[2]
            print(output_csv)
        except Exception as e:
            print("Not output csv specified - using default")
        if (os.path.isdir(path)):
            readwrite = process.Read_Write_Data(path)
            print("read in files")
            readwrite.read_in_files(2)
            print("done reading files")
            data = readwrite.get_df()
            print (data.head())
            cleaner = process.Clean_Data(data)
            print("clean data")
            cleaner.clean_data(textRank)
            print("done cleaning data")
            print("remove stop words")
            cleaner.remove_stop_words()
            print("done removing stop words")
            print("lemmatize")
            cleaner.lemmatization(True)
            print("done lemmatizing")
            if textRank:
                textrank = process.TextRank(readwrite.get_df())
                textrank.main()
            data = readwrite.get_df()
            print(readwrite.get_df())
            print("output to csv")
            readwrite.df_to_csv(output_csv)
            print("done all")
        else :
            print(path + " is not a valid directory")
    except Exception as e:
        print("Error running applicaton: " + str(e))

if __name__ == "__main__":
    main()
