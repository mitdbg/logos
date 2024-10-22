## Getting Started

### Installations

To run and contribute to the project, you will need [Git](https://git-scm.com/downloads), and [Python 3.10](https://www.python.org/downloads/) (which comes with [pip](https://pip.pypa.io/en/stable/)) installed on your computer. This project utilizes [Streamlit](https://docs.streamlit.io/) for the frontend and backend.

For Windows, you should download [Git Bash](https://gitforwindows.org/) for using the terminal.

For MacOS/Linux, you should already have a terminal pre-installed with the operating system.

If you have not already downloaded a text-based code editor, you should download [Visual Studio Code](https://code.visualstudio.com/).

Then, choose the directory where you want to clone the project repository.
```bash
# Choose the directory for the repository
$ cd PATH_TO_REPO

# Clone this repository
$ git clone @github.com:mitdbg/causal-log.git

# Go to the webapp directory
$ cd causal-log/webapp
```

### Python Packages

Once in the root directory, use the package manager pip to install [pipenv](https://pypi.org/project/pipenv/). 

```bash
$ pip install pipenv
```

Then activate a virtual environment and install the package dependencies.

```bash
$ pipenv shell
$ pipenv install
```

### Running the web server

On a separate, dedicated terminal with a virtual environment running (`pipenv shell`), go to the `webapp` directory and run

```bash
$ python3 -m streamlit run Home.py
```

Then, follow and click the link to the Network URL that appears on the terminal

### Project Structure

Below is a brief outline of the overall project file structure:
```
webapp/      
├── images
├── log_results  <--- output results for the current log
├── pages        <--- pages for the multipage app
│   ├── 1_📚_Demo_1.py
│   ├── 2_📝_Demo_2.py
│   ├── 3_🎓_Demo_3.py
│   └── 4_💨_Demo_4.py
├── Home.py      <--- main entry point to the app
└── sawmillUI.py <--- interacts with sawmill
```