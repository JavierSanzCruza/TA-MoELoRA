## Task-Aware MoELoRA for Universal Information Extraction

This repository contains code and supplementary materials to reproduce the experiments in the 
paper:

> L. Guo, J. Sanz-Cruzado, R. McCreadie. [Selecting the Right Experts: Generalizing Information Extraction for Unseen Scenarios via Task-Aware Expert Weighting](https://doi.org/10.3233/FAIA251308).
> 28th European Conference on Artificial Intelligence (ECAI 2025), Bologna, Italy, pp. 4161-4168.

The proposed model uses a Large Language Model (in our paper, [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf)) for extracting information from text. 
The original LLM is fine-tuned via a Mixture of LoRA experts (MOELoRA), with a task-aware router to
select the experts that combines (a) the token embeddings provided as input to each transformer block, 
and (b) a static embedding of the task using a pre-trained encoder (in our experiments, 
[CodeT5](https://huggingface.co/Salesforce/codet5-base)).

- **Paper:** [https://doi.org/10.3233/FAIA251308]()
- **Supplementary material:** [https://github.com/lubingzhiguo/TA-MoELoRA/blob/main/Appendix.pdf]()
- **GitHub repository:** [https://github.com/lubingzhiguo/TA-MoELoRA]()
- **HuggingFace model:** [https://huggingface.co/lbzg/TA-MoELoRA]()

## Authors
- **Lubingzhi Guo (corresponding author):** [l.guo.1@research.gla.ac.uk]()
- Javier Sanz-Cruzado: [javier.sanz-cruzadopuig@glasgow.ac.uk]()
- Richard McCreadie: [richard.mccreadie@glasgow.ac.uk]()

## Installation

The Task-Aware MoELoRa model in this repository is implemented based on [GoLLIE](https://github.com/hitz-zentroa/GoLLIE)
and [peft](https://github.com/huggingface/peft). We detail here how to install our models.

### Installing requirements

In order to execute the models included in this software, it is necessary to install the following packages:
(tested on Python > 3.10.)

- PyTorch > 2.2.0: `pip install torch`
- Transformers > 4.44.2: `pip install transformers`
- Accelerate > 0.34.2: `pip install accelerate`
- Bitsandbites > 0.43.3: `pip install bitsandbites`
- Black > 24.8.0: `pip install black`
- Datasets > 3.0.0: `pip install datasets`
- Flash attention > 2.5.8: `pip install flash-attn --no-build-isolation`

### Installing Task-Aware MoE LoRA model

In order to install the Task-Aware MoE LoRA model, it is necessary to first,
download the repository from GitHub and copy it into your local repository:

```bash
git clone https://github.com/lubingzhiguo/TA-MoELoRA.git
```

Once downloaded, the repository has the following structure:
```bash
repository
├── TA-MoELoRA
└── peft_tamoelora
└── notebooks
```

where `TA-MoELoRA` contains the model, `peft_tamoelora` is a custom version of the [peft](https://github.com/huggingface/peft)
library containing our implementation of the Mixture of Experts adapters and `notebooks` contains 
notebook examples that can be used to train and run the model.

In order to be able to use the Task-Aware MoE LoRA models, it is first necessary to install the `peft_tamoelora` package.
For this, use the following command.

```bash
cd peft_tamoelora
pip install -e .
```

## Pretrained Adapters

We release the trained moelora adapters based on [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf) with task encoder [CodeT5](https://huggingface.co/Salesforce/codet5-base) on HuggingFace. 
The adapters are available [here](https://huggingface.co/lbzg/TA-MoELoRA).

- **Model name:** `lbzg/TA-MoELoRA`
- **Base model:** [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf)

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

If you use the contents of this repository, please, cite the following paper:
```bibtex
@inproceedings{
    authors = { Guo, Lubingzhi and
                Sanz-Cruzado, Javier and
                McCreadie, Richard },
    title = {{Selecting the Right Experts: Generalizing Information Extraction for Unseen Scenarios via Task-Aware Expert Weighting}},
    booktitle = {28th European Conference on Artificial Intelligence (ECAI 2025)},
    pages = {4161--4168},
    address = {Bologna, Italy},
    publisher = {IOS Press},
    doi = {10.3233/FAIA251308}
}
```