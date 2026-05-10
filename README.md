# One Model, Many Geometries: Spectral Dynamics of Multilingual LLMs

## Group number
Our group is g33.

## Set-up

1. Create your own working subfolder on the cluster.

2. Copy the `.env.example` file and name it `.env`. Fill the fields inside it with your own details:
    - `HF_TOKEN`: Your Hugging Face token.
    - `GASPAR`: Your EPFL GASPAR username used for Run:AI.
    - `CLUSTER_FOLDER`: Your specific subfolder name on the team's scratch partition.
    - `GROUP`: Our group number (e.g., `g33`).

3. Copy the `.env` file to your subfolder on the cluster. 

4. Setup local python environment (only visualization of results):
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.local.txt
```

## Submitting jobs 

> ⚠️ **Warning:** Ensure you have created your own working subfolder on the cluster to avoid conflicts.

### Tracing geometry

For this type of jobs specifically, I have simplified the submission and tracking. 

To sumbit for Fuxi model: 

```
./job_scripts/trace_fuxi.sh <job-name>
```

To submit for Apertus:
```
./job_scripts/trace_apertus.sh <job-name>
```
Both scripts pull Gaspar, group, and cluster folder form the `.env` file.

