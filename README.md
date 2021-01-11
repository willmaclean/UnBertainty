# unBertainty


This is UnBertainty, a modular Python implementation of BioBert optimised to analyse uncertainty in the language of cardiac radiology. 

A revolution in radiology is underway thanks to machine learning. In a variety of different problem-settings, the texts of radiology reports need to be used to extract information or provide labels. However, hedging and uncertainty in radiology are significant problems for information extraction, since they can make it seem as if multiple conditions are present in a scan when in fact they may not be. UnBertainty is a step towards solving this problem, as it provides a soft flag for hedging for documents in a corpus, so that these flagged documents can be managed in an appropriate manner.

To learn more about the motivation for UnBertainty, refer to these slides: https://docs.google.com/presentation/d/1rMzCYhH7DIJQQp6hJSP0DKiEQnqyWQQC0457iTAuZsk/edit?usp=sharing

### What does it do?

UnBertainty processes Cardio-Magnetic Resonance imaging text reports and labels if they contain uncertain language. You can either fine-tune UnBertainty on CMR reports which you have already labelled as uncertain or not, or you can use the off-the-shelf model which has already been fine-tuned.

### What is UnBertainty NOT?

UnBertainty is NOT an infallible detection mechanism for uncertainty, and should be seen as a valuable soft flag as part of a wider processing pipeline for radiology reports. Uncertainty is a tricky concept to define, and so what might be flagged as uncertainty in some cases may not in others. World-knowledge and hospital context are key in this domain. Therefore, we judged it would be more useful to have a lightweight processor which could quickly identify potential uncertainty, rather than misleadingly claim to have an exhaustive method for uncertainty cleaning.

UnBertainty is trained and optimised to spot the presence of multiple diagnoses in a CMR report. No more, no less.

### Design

UnBertainty uses BioBert, which is a version of Google's massive language model trained on a biomedical corpus.

### Package Contents

(Only picking out the important bits.)

pipe.py: The pipeline for fine-tuning reports. Saves new model weights. to your current directory.

  --argv[1]: the directory containing CMR reports

tokenizer.py: The tokenizer which zeroes in on the conclusion section of the report if there is one.

label.py: If you want to label your reports with an off-the shelf model.

### Requirements 

1. HuggingFace Transformers, PyTorch, Keras, Tensorflow (2<)
2. CUDA
3. Biobert weights downloaded and accessible. Install Biobert here : https://github.com/dmis-lab/biobert

email willmaclean@gmail.com for problems.
