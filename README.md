# Extending the Scope of Out-of-Domain: Examining QA models in multiple subdomains

This is the repo containing the code for the paper in ACL 2022 Workshop on Insights from Negative Results in NLP titled "<em>Extending the Scope of Out-of-Domain: Examining QA models in multiple subdomains</em> ".


## 1. Download datasets
Download QA datasets from this url: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [NewsQA](https://github.com/mrqa/MRQA-Shared-Task-2019). The question classification data is in [https://cogcomp.seas.upenn.edu/Data/QA/QC/
](https://cogcomp.seas.upenn.edu/Data/QA/QC/).


## 2. Preprocessing

Using `preprocess.py` to transform SQuAD, NewsQA and question classification data to the form that huggingface can process.

## 3. Exploring out-of-subdomain performance of QA models
Firstly use the question classification data to train a question classifier using huggingface script [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py):
```
python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --train_file question_classification/train.csv \
  --validation_file question_classification/dev.csv \
  --test_file question_classification/test.csv \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir trained_models/bert-question-classification \
  --overwrite_output_dir
```

then classify the questions in SQuAD and NewsQA using the trained question classifier. 
```
python run_glue.py \
  --model_name_or_path trained_models/bert-question-classification \
  --train_file question_classification/train.csv \
  --validation_file question_classification/dev.csv \
  --test_file question-answering/newsqa/train_qtype.csv \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --output_dir /tmp/newsqa/ \
  --overwrite_output_dir \
  --overwrite_cache
  
python run_glue.py \
  --model_name_or_path trained_models/bert-question-classification \
  --train_file question_classification/train.csv \
  --validation_file question_classification/dev.csv \
  --test_file question-answering/squad_1.1/train_qtype.csv \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --output_dir /tmp/squad_1.1/ \
  --overwrite_output_dir \
  --overwrite_cache
```

Second, use the functions in `export_by_subdomain.py` to export QA examples by subdomains (question type, text length and answer position) using increasing data size with specified intervals (using text length as an example in `export_by_subdomain.py`):

```
python export_by_subdomain.py
```

Finally, you can train QA models on these subsets of the original QA datasets then evaluate QA models on the dev set using huggingface script [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py):

```
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --train_file  question-answering/squad_1.1/sub_datasets/length/squad_1.1_train_answer_long_500.json \
  --validation_file  question-answering/squad_1.1/dev.json \
  --do_train  \
  --do_eval \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --per_device_eval_batch_size=12 \
  --per_device_train_batch_size=48 \
  --overwrite_output_dir \
```

Using `evaluate_qa.py` to evaluate all QA systems trained on subsets of the original dataset (using text length as an example in `evaluate_qa.py`):

```
python evaluate_qa.py
```

## 4. Visualizing the results

After evaluation, you can use the functions in `visualization.py` to visualize the experimental results of the effects of subdomains (using text length as an example in `visualization.py`):

```
python visualization.py
```

