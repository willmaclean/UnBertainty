def tokenize(reports):
    """
    Tokenizes and masks reports.

    Accepts:

    The cleaned texts.

    Returns:
    attention_masks
    input_ids (sequences of tokenized words)

    Biobert and keras must be installed

    """
    import torch
    from transformers import BertTokenizer
    from keras.preprocessing.sequence import pad_sequences

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)

# Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    sentences = reports.texts.values
    
    conc_loc = 0
    for i in range(len(sentences)):
        if 'conclusion' in sentences[i]:
            conc_loc = i
    
    #splitting the report for where there is a conclusion if there is one
    sentences = sentences[conc_loc:]
    
# For every sentence..
    for sent in sentences:
        encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP
                   )

    # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    #PADDING
    # Set the maximum sequence length.
    # Max length is the length of longest report
    lens = reports.texts.map(lambda x: len(x.split(' ')))
    MAX_LEN = max(lens)


    # Pad our input tokens with value 0.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

#masking
    attention_masks = []

    for sent in input_ids:
    #    token ID is 0, then it's padding, set the mask to 0.
    #   token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return input_ids, attention_masks
