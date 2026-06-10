# One Model, Many Geometries: Spectral Dynamics of Multilingual LLMs

Please, follow the instructions below to reproduce all of our experiments. We set up everything to ensure multiple people can run the code on the cluster without conflicts.

## Set-up

1. Create your own working subfolder on the cluster within `scratch` and clone the repository there.

2. Copy the `.env.example` file and name it `.env`. Fill the fields inside it with your own details:
    - `HF_TOKEN`: Your Hugging Face token, **mandatory** to access gated models.
    - `GASPAR`: Your EPFL GASPAR username used for Run:AI.
    - `CLUSTER_FOLDER`: Your specific subfolder name on the team's scratch partition.
    - `GROUP`: Our group number (e.g., `g33`).
    - `OPENAI_API_KEY`: An OpenAI API key for LLM judges (only needed for multilingual transfer experiments, skip not needed).
    - `CSCS_SERVING_API`: A [Swiss AI Research Platform](https://serving.swissai.svc.cscs.ch/) API key for LLM judges (only needed for multilingual transfer experiments, skip not needed).

3. Copy the `.env` file to your subfolder on the cluster. 

## Submitting jobs 

> ⚠️ **Warning:** Ensure you have created your own working subfolder on the cluster to avoid conflicts.

### Tracing Geometry over Pretraining

To submit the jobs for RankMe computation over all checkpoints, use the following scripts, which pull Gaspar, group, and cluster folder form the `.env` file:

```
./code/job_scripts/trace_fuxi_fine_wiki.sh <job-name>
./code/job_scripts/trace_apertus_fine_wiki.sh <job-name>
```

### Multilingual Transfer

#### ECLeKTic

To generate answers, scores, and correlation scores for ECLeKTic, follow these steps:

1. Add `OPENAI_API_KEY` and `CSCS_SERVING_API` to `.env` if not already done.
2. Submit the job for answer generation and scoring (both models at the same time):
```
./code/job_scripts/eclektic.sh
```
3. Run the correlation analyses using the following bash script (both models at the same time):
```
./code/multilingual_transfer/scripts/run_correlations_eclektic.sh
```

#### XNLI

To generate answers and correlation scores for the XNLI setting, follow these steps:

1. Submit the generation and scoring jobs for each model (can be run simultaneously):
```
./code/job_scripts/xnli_apertus.sh <job-name>
./code/job_scripts/xnli_fuxi.sh <job-name>
```
2. Run the correlation analyses (both models at the same time):
```
./code/multilingual_transfer/scripts/run_correlations_xnli.sh
```
3. Run the correlation analyses with RankMe law predictors:
```
./code/multilingual_transfer/scripts/run_correlations_xnli_law.sh
```

### Downstream Monolingual Performance

Instructions for submitting downstream evaluation jobs (m-MMLU, XCOPA, Belebele, M-ARC) are in [`code/downstream_evaluation/README.md`](code/downstream_evaluation/README.md).