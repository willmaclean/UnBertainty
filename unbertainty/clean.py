

class TextPreprocessor:

    """
    For cleaning the texts.
    """

    def __init__(self):
        pass

    def clean(self, text):
        import re
        text = re.sub('\(.*\)', '', text)
        text = re.sub('/\r?\n|\r/', '', text)
        text = re.sub("\d+", "", text)
        return text

    #splitting the conclusions from the reports. This is an unsupported option
    def get_conclusions(self, text):

        #split each report
        sections = text.split('\n')

        #go to the last element in the list which has text
        conclusion = ''
        for i in range(len(sections)-1, 0, -1):
            if len(sections[i])>3:
                conclusion = sections[i]
                break
        return conclusion

    def get_conc_length(self, text):

        return len(text)
