# unBertainty (unfinished, work in progress, lots of bugs )


This is UnBertainty, a set of tools written in python for handling uncertainty in radiology reports.

It also includes a script for fine-tuning BioBert on your own suite of reports and labelling them as uncertain or not.

A revolution in radiology is underway thanks to machine learning. In a variety of different problem-settings, the texts of radiology reports need to be used to extract information or provide labels. However, hedging and uncertainty in radiology are significant problems for information extraction, since they can make it seem as if multiple conditions are present in a scan when in fact they may not be.

UnBertainty is a step towards solving this problem, as it provides a soft flag for hedging for documents in a corpus, so that these flagged documents can be managed in an appropriate manner.

### What does it do?

Unbertainty has some tools for handling radiology reports like uncertainty detection, conclusion splitting, and tokenisation.

UnBertainty has a rules-based uncertainty detection mechanism, using uncertainty cues farmed from the academic literature.

It also uses BioBert to perform machine-learning based uncertainty detection. You can either fine-tune UnBertainty on CMR reports which you have already labelled as uncertain or not, or you can use the off-the-shelf model which has already been fine-tuned.

### What is UnBertainty NOT?

UnBertainty is NOT an infallible detection mechanism for uncertainty, and should be seen as a valuable soft flag as part of a wider processing pipeline for radiology reports. Uncertainty is a tricky concept to define, and so what might be flagged as uncertainty in some cases may not in others. World-knowledge and hospital context are key in this domain. Therefore, we judged it would be more useful to have a lightweight processor which could quickly identify potential uncertainty, rather than misleadingly claim to have an exhaustive method for uncertainty cleaning.

UnBertainty is trained and optimised to spot the presence of multiple diagnoses in a CMR report. No more, no less.

### Design

For rules-based labelling, UnBertainty uses uncertainty cues from responses to the 2010-ConLL shared task (Farkas et al., 2010) and NegBio (Peng et al. 2018). 

For ML labelling, UnBertainty uses BioBert, which is a version of Google's massive language model trained on a biomedical corpus.

### Requirements 

1. HuggingFace Transformers, PyTorch, Keras, Tensorflow (2<)
2. CUDA
3. Biobert weights downloaded and accessible. Install Biobert here : https://github.com/dmis-lab/biobert

email willmaclean@gmail.com for problems.
