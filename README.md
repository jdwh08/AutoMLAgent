# AutoML Agent

Prototype modular, agentic AutoML system for (tabular datasets for now).

[![CI](https://github.com/jdwh08/AutoMLAgent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jdwh08/AutoMLAgent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdwh08/AutoMLAgent/branch/main/graph/badge.svg)](https://codecov.io/gh/jdwh08/AutoMLAgent)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Research Question**: Can agents achieve similar levels of performance as
manual model training on benchmark /  Kaggleish datasets?

- How powerful does the model need to be? Can we keep this local (or local on a non-potato)?
- Can this be achieved with tool calls alone, or do we need code excution (security is questionable)?
- Do we need to use graphs? If yes, how sophisticated does the agent process need to be?

## Sketch

Testing benchmark for agentic capabilities.

1. AutoEDA (WIP) -- we assume tabular datasets for now. How far can we get with EDA as tools?
2. AutoFE -- is it possible to reasonably constrain feature engineering tasks to as individual tools (e.g., a BoxCox tool, a SMOTE tool, etc.), or does this require more sophisticated treatment with code execution?
3. AutoML -- can notes alone be enough to select good models? Does adding a LLM enable more efficient traversal of the model search space (e.g., go immediately to Gradient-Boosted Trees for tabular data), or does it prevent exploration?
4. AutoHPT -- this is kinda old hat with stuff like Optuna or other bayesian-based hyperparameter tuning.
5. AutoEvaluation -- how does the agent perform in terms of metrics?
6. AutoWriteup -- these are language models, right? Surely they can write documentation.

## Tech Stack

### Core Components

- Data ingestion & validation (Pydantic v2 models)
- API-based LMs to start with, not because I like it but because I don't have a nice workstation for research yet and am currently working on my potato that I have a soft spot for.
- Pydantic AI Agents ([docs](https://ai.pydantic.dev/agents/)), leveraging type-safe, composable agents for each pipeline stage.
  - If necessary, graph-based with LangGraph or other features.
- MLflow. Reluctantly for the LLM logging side. Has a bonus of being local only.
- pytest (unit & integration tests)
- Ruff, UV

### Agents

- Exploratory Data Analysis (EDA) agent
- Feature engineering agent
- Model selection & training agent
- Hyperparameter optimization agent
- Evaluation/reporting agent
- Orchestration agent to coordinate workflow

### Configuration

- pyproject.toml

## Datasets

For now, we focus only on non-competition tabular kaggle datasets which I remember my experience with (the "training wheels"). In alphabetical order:

- [ ] Credit Card Approval Dataset <https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction>
- [ ] Credit Risk Dataset <https://www.kaggle.com/datasets/laotse/credit-risk-dataset>
- [ ] COVID-19 <https://www.kaggle.com/datasets/georgesaavedra/covid19-dataset>
- [ ] House Rent Prediction <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>
- [ ] Flu Shot <https://www.drivendata.org/competitions/66/flu-shot-learning/data/>
- [ ] Titanic Survival (everyone's favourite hello world) <https://www.kaggle.com/c/titanic/data>
- [ ] Long Term TODO: Automatically download and process more ml competition datasets (e.g., create, find, and run `jdwh08/get-comps-data`)

## Papers and References to Consider

### General

- AutoML <https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf>

### EDA

- <https://github.com/mstaniak/autoEDA-resources>
  - DataPrep EDA <https://arxiv.org/pdf/2104.00841>
  - Smart EDA <https://arxiv.org/pdf/1903.04754>

### Feature Engineering

- Feature Engine <https://feature-engine.trainindata.com/en/latest/index.html>
- Feature Tools <https://github.com/alteryx/featuretools>

### Model Selection

- NNI for NN <https://github.com/microsoft/nni>
- Auto sklearn <https://github.com/automl/auto-sklearn/tree/development>
- TPOT <https://github.com/EpistasisLab/tpot>
- AILink <https://github.com/alibaba/Alink>
- MLJar <https://github.com/mljar/mljar-supervised>

## Disclaimers and License

```text
　　　　　 　 ____
　　　　　／＞　 　フ
　　　　　|   _　 _|   *sigh*
　 　　　／` ミ＿xノ
　　 　 /　　　 　 |
　　　 /　 ヽ　　 ﾉ
　 　 │　　|　|　|
　／￣|　　 |　|　|
　| (￣ヽ＿ )__)__)
　 ＼二つ 
```

### Particularly Important Disclaimers

This project is a proof of concept purely for research purposes
and not intended for commercial, production, clinical, or other use.

The opinions, datasets, and methodologies expressed or used herein are solely my own
or are publicly available, and do not reflect the views or opinions of my employer.

This project does not guarantee any particular results, accuracy, or performance.
The resulting models are not intended to be used as professional advice in any way.

While metrics may be included, users are strongly encouraged to validate on their datasets.
The replication crisis is real.
Testing on your own data is particularly needed when considering real world risks
such as bias, adversarial datasets, data compliance, etc.

### License (and Additional Disclaimers)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Go open!
