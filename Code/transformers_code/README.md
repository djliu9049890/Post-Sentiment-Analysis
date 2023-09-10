# <a name="robotarium_webapp"></a>Hugging Face Transformers Playground

We will run [Hugging Face transformers models](https://huggingface.co/docs/transformers/index) using [PyTorch](https://pytorch.org/get-started/locally/). [BART](https://huggingface.co/docs/transformers/model_doc/bart#bart) model will be focused here, but other models are also explored for various tasks such as sentiment classification, sentence prediction, text generation and zero shot classification. Models for Chinese language and fine tuning a pretrained model are also explored.

Note that for the playground here, [PyTorch](https://pytorch.org/get-started/locally/) is installed instead of [TensorFlow](https://www.tensorflow.org/install/pip) or [Flax](https://flax.readthedocs.io/en/latest/) for the deep learning library dependency. If [TensorFlow](https://www.tensorflow.org/install/pip) or [Flax](https://flax.readthedocs.io/en/latest/) is desired, please visit [Hugging Face Installation Guide](https://huggingface.co/docs/transformers/installation) to install and use other options.

## <a name="installing_getting_started"></a>Installing / Getting Started
### <a name="prerequisites"></a>Prerequisites

  - Install [Miniconda](https://conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary) or [Anaconda](https://www.anaconda.com/) for the `conda` commands
    - Follow the installation guide here to install: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
    - Note: It is recommended to install [Miniconda](https://conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary) with the link above instead of [Anaconda](https://www.anaconda.com/) if you did not already have [Anaconda](https://www.anaconda.com/) installed
  - Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to clone Github repositories

### <a name="hugging_face_env"></a>Create the `hugging-transformers` virtual environment

Follow the steps to create the `hugging-transformers` env with [PyTorch](https://pytorch.org/get-started/locally/):

  1. Open a new terminal, run `conda create -n hugging-transformers python=3.10`
  2. Run `conda env list` to verify that the new environment was installed
  3. Run `conda activate hugging-transformers` to activate the `hugging-transformers` virtual environment
  4. Run `pip3 install -r requirements.txt` to install necessary libraries for the environment
  5. Run `conda deactivate` to deactivate the `hugging-transformers` virtual environment
  6. Ensure the virtual environment is now `base` on the terminal

  NOTE: To update the `hugging-transformers` virtual environment when there is a change in the `requirements.txt` file, run `pip install --upgrade -r requirements.txt` on the `hugging-transformers` environment to update.

### <a name="verify_pytorch"></a>Verify PyTorch is installed correctly

1. Execute `conda activate hugging-transformers` on a terminal
2. Execute `python` and run the following commands in Python:

    ```python
    import torch
    x = torch.rand(5, 3)
    print(x)
    ```

    The output x will look like below:

    ```python
    tensor([[0.3380, 0.3845, 0.3217],
            [0.8337, 0.9050, 0.2650],
            [0.2979, 0.7141, 0.9069],
            [0.1449, 0.1132, 0.1375],
            [0.4675, 0.3947, 0.1426]])
    ```

    Visit [PyTorch](https://pytorch.org/get-started/locally/) official website for more detail.

### <a name="verify_transformers"></a>Verify Transformers is installed correctly

1. Run `conda activate hugging-transformers` on a terminal
2. Run the following command:

   `python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"`

   The output will look like below:

   `[{'label': 'POSITIVE', 'score': 0.9998704791069031}]`

## <a name="run_script"></a>Running a Python Script

Run one of the Python scripts in the directory by:

1. Open a new terminal and cd to the `bart/` directory
2. Run `conda activate hugging-transformers` on the terminal
3. Run `python <filename.py>` to execute the Python script called `filename.py`
    * For example: `python bart_summarize.py`