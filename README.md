# LOGos

Utilizing system logs to perform causal analysis. You can access the documentation [here](https://mitdbg.github.io/logos). 

### OpenAI integration

In order to use the LLM-powered capabilites of LOGos, please add a `.env` file to the root of this repo and define `OPENAI_API_KEY` appropriately.

### Reproducing our evaluation

To reproduce the evaluation from our VLDB paper, please follow the following steps:

1. Follow the instructions in `dataset_files/README.md` to gain access to our datasets.
2. Within `evaluation/`, you will find directories based on each experiment presented in our paper. Based on the experiment you would like to reproduce, switch into the appropriate directory and run the `reproduce.sh` script (you may need to edit file permissions to make it executable). The results will be saved under `evaluation/repro_results/` and the corresponding plots under `evaluation/repro_plots`

### Demos

You can find a quick demo of the LOGos API at [`demo/demo.ipynb`](demo/demo.ipynb).
