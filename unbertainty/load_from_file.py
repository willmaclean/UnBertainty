def load_from_dir(directory):

    """
    Loads .txt files and returns a dataframe
    """

    from os import listdir
    from os.path import isfile, join
    import pandas as pd

    try:

        file_names = [f for f in listdir(directory) if isfile(join(directory, f))]
        texts=[]
        ids=[]
        for file_name in file_names:

            try:

                file = open(directory+'/'+file_name)
                text = file.read()
                texts.append(text)
                ids.append(file_name[:-4])

            except:
                pass

    except ValueError:
            print('The path you entered does not exist.')

    df = pd.DataFrame({'id':ids,'texts': texts, })

    return df
