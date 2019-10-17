import sys
# import preprocessing as pre


def main():
    # path = './cnn'
    try:
        path = sys.argv[1]
        print(path)
        # process = pre.Clean_Data(path)
        # process.clean_data()
        # process.remove_stop_words()
    except Exception as e:
        print(e)


main()
