## Task-Aware MoELoRA for UIE

Code and supplementary materials for "Selecting the Right Experts: Generalizing Information Extraction for Unseen Scenarios via Task-Aware Expert Weighting" (ECAI 2025)


## Installation

Task-Aware MoELoRA is implemented based on [GoLLIE](https://github.com/hitz-zentroa/GoLLIE) and [peft](https://github.com/huggingface/peft). 
```bash
conda create -n moelora python=3.10
conda activate moelora

git clone https://github.com/lubingzhiguo/TA-MoELoRA.git
cd TA-MoELoRA
pip install -r requirements.txt
cd peft
pip install -e .
```

## Pretrained Adapters
We release the trained moelora adapters based on [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf) with task encoder [CodeT5](https://huggingface.co/Salesforce/codet5-base). The adapters are available [here](https://huggingface.co/lbzg/TA-MoELoRA).

## Generate Dataset 
Please refer to the [Generate the GoLLIE dataset](https://github.com/hitz-zentroa/GoLLIE?tab=readme-ov-file#generate-the-gollie-dataset) for detailed dataset collection and generation instructions. For the SciERC, SemEval, ADECorpus, CoNLL2004, NYT11-HRL, KBP37, GIDS, and PHEE datasets, we use the text splits provided by [IEPile](https://github.com/zjunlp/IEPile).

## Training and Evaluation
```
CONFIGS_FOLDER="configs/model_configs"
python3 -m src.run ${CONFIGS_FOLDER}/moelora_task_aware.yaml
```
To evaluate the pretrained adapters, add the following line to the configuration file.
```
lora_weights_name_or_path: lbzg/TA-MoELoRA
```

## Citation