<<<<<<< HEAD
# nlpipe

## Overview

This project is meant to be a one stop shop for all the natural language processing needs. It is intended to put together the state of the art NLP models just a step away from integration. Just the idea of whiteboard to production excits us. The library now consists of state of the art vectorization methods and industrial standard text analytics. 

Nodes Present in this version of nlpipe:

1. Data Engineering:
   1. Load data
   2. Split incoming data into test and train sets
2. Data Preprocessing:
   1. Spacy Tokenization (custom sentence tokenizer available) 
   2. Spacy Dependency Parser
   3. Spacy Parts of Speech Tagger
3. Vectorization:
   1. Corex Topic Modeling 
   2. BERT sentence vectorizer
   3. Auto encoders to combine various vectors
   4. LSTM encoders to combine various vectors
4. Actionables:
   1. GMM Models
   2. GradientBoost Models
   3. Umap interactive plots
5. Summarization.
   1.Page Ranking 
 
An example pipeline was integrated using the library to classify steam reviews. We can leverage a very powerful tool which allows us to see how each nodes are connected by using kedro-viz
 
  ![Kedro_pipeline](/kedro-pipeline-2.svg)
 

 
 
## Setup/Install dependencies

1. Install anaconda and create a conda virtual environment using 
```
conda create --name NLP
```
2. Activate the virtual environment using the following command

```
conda activate NLP
```

3. Install the dependencies using requirements.txt (pip freeze of my project workspace)

```
pip install -r requirements.txt
```





Take a look at the [documentation](https://kedro.readthedocs.io) to get started.

## Guidelines

In order to get the best out of the template:

 * Please don't remove any lines from the `.gitignore` file we provide
 * Make sure your results can be reproduced by following a data engineering convention, e.g. the one we suggest [here](https://kedro.readthedocs.io/en/stable/06_resources/01_faq.html#what-is-data-engineering-convention)
 * Don't commit any data to your repository
 * Don't commit any credentials or local configuration to your repository
 * Keep all credentials or local configuration in `conf/local/`


## Running Kedro

You can run your Kedro project with:

```
kedro run
```



## Working with Kedro from notebooks

In order to use notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

For using Jupyter Lab, you need to install it:

```
pip install jupyterlab
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

You can also start Jupyter Lab:

```
kedro jupyter lab
```

And if you want to run an IPython session:

```
kedro ipython
```

Running Jupyter or IPython this way provides the following variables in
scope: `proj_dir`, `proj_name`, `conf`, `io`, `parameters` and `startup_error`.

### Converting notebook cells to nodes in a Kedro project

Once you are happy with a notebook, you may want to move your code over into the Kedro project structure for the next stage in your development. This is done through a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#cell-tags) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`.
```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. To this end, you can run the following command to convert all notebook files found in the project root directory and under any of its sub-folders.
```
kedro jupyter convert --all
```

### Ignoring notebook output cells in `git`

In order to automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be left intact locally.

## Package the project

In order to package the project's Python code in `.egg` and / or a `.wheel` file, you can run:

```
kedro package
```

After running that, you can find the two packages in `src/dist/`.

## Building API documentation

To build API docs for your code using Sphinx, run:

```
kedro build-docs
```

See your documentation by opening `docs/build/html/index.html`.

## Building the project requirements

To generate or update the dependency requirements for your project, run:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.



______________________________________________________________________________________________________________________________
______________________________________________________________________________________________________________________________
______________________________________________________________________________________________________________________________

This package provides essential elements of nlp project. The current project is in development phase. We hope to maintain state of the art and widely used nlp approaches integrated to respective landscape we defined. Please contribute your new nlp approaches and we would love to integrate your algorithms as we see fit.


The example pipeline we have here follows the flow/pipeline figure as seen in kedro-pipeline-2.svg











