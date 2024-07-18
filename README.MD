*Parsing and Data Collection for Code4MLv2 Dataset*
[![DOI](https://zenodo.org/badge/829904747.svg)](https://zenodo.org/doi/10.5281/zenodo.12772256)


This collection of scripts and notebooks aims to extract and gather different characteristics and background information from Kaggle notebooks. 

**Files and Descriptions**
1. parse_comps.ipynb
    - Purpose: Filters and collects notebooks that will be used in subsequent parsing steps.
    - Output: This notebook produces output needed for meta_kaggle_codeparser.ipynb.
2. meta_kaggle_codeparser.ipynb
    - Purpose: Parses the chosen notebooks to extract specific code components.
    - Output: The output from this notebook is required for parse_graphs.ipynb.
3. parse_graphs.ipynb
    - Purpose: Collects features from the parsed notebooks.
    - Output: This notebook's output is needed for get_data_type.py.
4. get_data_type.py
    - Purpose: Collects data types from the code using a Large Language Model (LLM).
    - Output: The output from this script is required for get_description.py.
5. get_description.py
    - Purpose: Collects descriptions from the code using a Large Language Model (LLM).
    - Output: The output is needed for code_competitions.ipynb.
6. code_competitions.ipynb
    - Purpose: Creates the competitions_meta2.csv file for the Code4MLv2 dataset.
    - Output: This notebook generates competitions_meta2.csv.
7. kg_meta.ipynb
    - Purpose: Creates the kernels_meta2.csv and code_blocks2.csv files for the Code4MLv2 dataset.
    - Output: This notebook generates kernels_meta2.csv and code_blocks2.csv.

## Setting Up the Environment

1. **Set OPENAI_API_KEY:**
   - Create a `.env` file in the root directory of your project.
   - Add your OpenAI API key to the `.env` file:

     ```bash
     OPENAI_API_KEY=your_openai_api_key_here
     ```

2. **OpenAI API:**
   - The scripts use GPT-3.5 and GPT-4 models via the OpenAI API.
   - Ensure you have the correct OpenAI version installed:

     ```bash
     openai==0.28.0
     ```
