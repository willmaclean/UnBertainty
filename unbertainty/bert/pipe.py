"""

The main pipeline. This is actually meant to be run as a script, so is not really part of the main module.

Steps:

1.load texts from file
2. clean
3. tokenize
4. Fine-tune BERT
5. save params and final hidden layer to file

args:

argv[1] = the name of the file with texts. NOT the path. 

"""
import sys
import load_from_file
from clean import TextPreprocessor
import tokenizer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == '__main__':

    #Load files

    reports = pd.read_csv(sys.argv[1])
    sys.stdout.write('     Starting the SCRIPT     ')
    #clean the reports
    prep = TextPreprocessor()
    reports['texts'] = reports['texts'].apply(prep.clean)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #tokenizing
    input_ids, attention_masks = tokenizer.tokenize(reports)

    #getting the labels. My labels are in the 'uncertain' column
    labs = reports[['uncertain']].values

    #Train test split
    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labs,
                                                            random_state=42, test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labs,
                                             random_state=42, test_size=0.1)

    #converting the data to torch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


    batch_size = 32


    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(
    "biobert_v1.1_pubmed", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = True, # Whether the model returns attentions weights.
    output_hidden_states = True)

    model.cuda()

    #for weight decay fixing and learning rate
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    from transformers import get_linear_schedule_with_warmup
    import time

    epochs = 4

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    #save the params
    params = list(model.named_parameters())
    # f_ = open('parameters.txt', 'w')
    # f_.write(params)


    with open("params.txt", "w") as output:
        output.write(str(params))

    import random


    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):


        # Perform one full pass over the training set.

        sys.stdout.write('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        sys.stdout.write('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. 
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                sys.stdout.write('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # clearing previously calculated gradients
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch
            outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels, return_dict=True)

           
            attention_output = outputs.attentions
            logits = outputs.logits

            # adding loss to loss over all batches
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # set the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        sys.stdout.write("")
        sys.stdout.write("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. 
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        sys.stdout.write("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        sys.stdout.write("  Validation took: {:}".format(format_time(time.time() - t0)))


    sys.stdout.write("Training complete!")

    import os

    output_dir = 'saved_models'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sys.stdout.write("Saving model to %s" % output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
