# Evaluation Guidelines
We provide detailed instructions for evaluation. 
To execute our evaluation script, please ensure that the structure of your model outputs is the same as ours.

## Data Loading, Model Inference, and Output Saving

We provide one example file `test_benchmark.py` to test the benchmark for your reference. 
Basically, it
- loads the dataset
- conducts model inference(*InternVL.py, Janus.py, Llama.py, Llava.py, llava-onevision.py, phi3.5.py, Qwen2.py, and Qwen2_5.py*))
- extracts the answer choices from model outputs (supported in `extrac_answer.py`)
- evaluate the final accuracy (supported in `evaluate.py`)
