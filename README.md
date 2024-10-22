# LOGos

Utilizing system logs to perform causal analysis. You can access the documentation [here](https://mitdbg.github.io/logos).

Please begin by installing the Python packages required for this project by running `pip install -r requirements.txt`.

### OpenAI integration

In order to use the LLM-powered capabilites of LOGos, please add a `.env` file to the root of this repo and define `OPENAI_API_KEY` appropriately.

### Trying out LOGos

For an introduction to our Python-based interface, you can turn to our demo notebook at [`demo/demo.ipynb`](demo/demo.ipynb).

We also offer a simple UI built using [Streamlit](https://docs.streamlit.io/). You can launch it by running [`demo/run_ui_demo.sh`](demo/run_ui_demo.sh) and following the resulting URL.


### Reproducing our evaluation

To reproduce the evaluation from our VLDB paper, please follow the following steps:

1. Follow the instructions in `dataset_files/README.md` to gain access to our datasets.
2. Within `evaluation/`, you will find directories based on each experiment presented in our paper. Based on the experiment you would like to reproduce, switch into the appropriate directory and run the `reproduce.sh` script (you may need to edit file permissions to make it executable). This will run the experiment and plot the results.
3. Find the resulting plots in `evaluation/repro_plots/`. The raw data for each plot will be saved in `evaluation/repro_plots_data/`.

