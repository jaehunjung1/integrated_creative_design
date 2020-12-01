## Question Generation for MRC

This is a project repository for Integrated Creative Design, Fall 2020.  
Below, we explain how to augment QA data from insurance document (stored in `dataset/insurance`), and train QA model on the augmented dataset.

## 1. Answer Span Selection
(1) `cd ner; pip install -r requirements.txt`  
(2) Download pretrained [NER model](https://drive.google.com/open?id=1FDLe3SUOVG7Xkh5mzstCWWTYZPtlOIK8) into `ner/experiments/base_model_with_crf`.  
(3) Run `python insurance_pipeline.py` to create preprocessed paragraphs / cloze sequences / answers from the insurance documents.

## 2. Question Generation
(1) `cd UnsupervisedMT/NMT`  
(2) Download [data.zip](https://drive.google.com/file/d/1sG4TdWD8BTRLulkKrV18I2S0UaOXBhUy/view?usp=sharing), and unzip it into `mono_qa_new` and `para_qa_new` folders. Move the two folders into `UnsupervisedMT/NMT/data/`.  
(3) Download pretrained [UNMT model](https://drive.google.com/file/d/1q-OqeQj_2NR5C_0wysFn82egUVkP8-8J/view?usp=sharing), and unzip it into `dumped` folder. Move the folder into `UnsupervisedMT/NMT/`.  
(4) Preprocess the cloze sequence file by running `./get_data_insurance_inference.sh`.  
(5) Translate from cloze to question by running `./insurance_inference.sh`. This will generate output question file in `UnsupervisedMT/NMT/data/clozes`.

## 3. Putting all together: Creating SQuAD-style dataset
(1) `cd {Project base folder}`  
(2) Run `python insurance_generate_squad.py`. This will generate SQuAD-style dataset in `dataset/KorQUAD/augmented_insurance.json`.  




## References
Garcia-Duran. A, Dumancic. S, and Niepert. M, Learning Sequence Encoders for Temporal Knowledge Graph Completion. EMNLP 2018
