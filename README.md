# Monitoring Discounted-Sum Properties

This repository contains the code to reproduce the experiments in the paper "Monitoring Discounted Sum Properties".

The repository uses the directory name `experimental-results/`. In the text
below, "result data" means the CSV files in that directory, and "images" means
the PDF plots in `experimental-results/images/`.

## External Data and Code

The raw MNIST dataset is public and is not meant to be committed to this
repository. The MNIST scripts use the standard `torchvision` layout under
`data/MNIST/`; download it with:

```bash
python -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True); datasets.MNIST(root='./data', train=False, download=True)"
```

The fairness model code in `ffb/` comes from
[ahxt/fair_fairness_benchmark](https://github.com/ahxt/fair_fairness_benchmark)
and is kept in this repository so the experiments are self-contained.

The FFB raw datasets are not meant to be committed. See
`ffb_datasets/readme.md` for the download link and expected directory layout.

## Training FFB ML Models

Figures 2 and 4 use the Adult/race/HSIC model
`model_adult_race_hsic.pkl`. If `experimental-results/MLmodels/` does not
already contain it, train it with the vendored FFB code.

First download the FFB datasets as described in `ffb_datasets/readme.md`. The
upstream FFB scripts expect datasets at `../datasets/...` when run from the
`ffb/` directory, so create a local symlink from `datasets/` to
`ffb_datasets/`:

```bash
test -e datasets || ln -s ffb_datasets datasets
mkdir -p experimental_results/ML_models experimental-results/MLmodels
```

Train the model used by the paper figures:

```bash
cd ffb
python ffb_tabular_hsic.py --dataset adult --target_attr income --sensitive_attr race
cd ..
cp experimental_results/ML_models/model_adult_race_hsic.pkl experimental-results/MLmodels/model_adult_race_hsic.pkl
```

The FFB scripts write trained models and logs to
`experimental_results/ML_models/`. The trace-generation script in this
repository reads from `experimental-results/MLmodels/`, so the copy command
above places the model where `generate_ffb_traces.py` expects it.

To train every FFB model configuration included in `ffb/run_all.sh`, run:

```bash
mkdir -p experimental_results/ML_models
cd ffb
bash run_all.sh
cd ..
```

That full run trains multiple fairness methods, datasets, and sensitive
attributes in parallel, so it is substantially heavier than the single HSIC
model needed for Figures 2 and 4.

## Reproducing Figures 1-4

This is the main paper-artifact workflow. The other plotting/table functions in
the repository are not needed for Figures 1-4.

### 1. Set up Python

Create and activate a local virtual environment. The `venv/` directory is
ignored by git:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install numpy pandas matplotlib tqdm scikit-learn scipy statsmodels tabulate torch torchvision seaborn
```

### 2. Obtain the result CSVs in `experimental-results/`

The figure CSVs are monitor outputs produced from the source traces and setup
JSON files. The repository already contains precomputed CSVs; to regenerate
them, first make sure the source traces below exist:

- Power trace input for Figure 1: `google-trace-data/cella_pdu6.csv`
- Adult fairness trace input for Figures 2 and 4:
  `ffb_traces/adult_race_hsic.csv`
- MNIST drift traces for Figure 3: `mnist-traces/mnist_drift_trace_*.csv`

No script in this repository creates `google-trace-data/cella_pdu6.csv`; place
that CSV at the path above before regenerating Figure 1 data.

The Adult and MNIST traces can be regenerated from the included models and raw
datasets:

```bash
python generate_ffb_traces.py
python generate_mnist_traces.py
```

`generate_ffb_traces.py` uses
`experimental-results/MLmodels/model_adult_race_hsic.pkl` and the Adult data in
`ffb_datasets/adult/raw/`. Download the FFB raw datasets as described in
`ffb_datasets/readme.md` and train the Adult/race/HSIC model as described in
[Training FFB ML Models](#training-ffb-ml-models) before regenerating this
trace.
`generate_mnist_traces.py` uses `experimental-results/MLmodels/mnist_cnn.pt`;
if you want to retrain that model first, run `python train_mnist.py`. The MNIST
data must be available in the `torchvision` layout under `data/MNIST/`; use the
download command in [External Data and Code](#external-data-and-code) if it is
missing.

Regenerate the experiment setup files for the power-trace and Adult experiments
when needed:

```bash
python generate_experimental_setups.py
```

Then run the monitor experiments for the paper figures. By default,
`main.py` runs all Figure 1-4 experiments:

```bash
python main.py
```

This is equivalent to:

```bash
python main.py run_all
```

To run only selected experiments, pass one or more experiment names:

```bash
# Figure 1 data
python main.py powertrace_decrease_eps powertrace_decrease_interval_length

# Figure 2 data
python main.py adult_decrease_eps

# Figure 3 data
python main.py mnist_increase_noise

# Figure 4 data
python main.py adult_decrease_interval_length
```

List the available experiment names with:

```bash
python main.py --list
```

These commands write:

| Figure | Result CSVs |
| --- | --- |
| Figure 1 | `experimental-results/powertrace_eps_*.csv`, `experimental-results/powertrace_interval_length_*.csv` |
| Figure 2 | `experimental-results/adult_race_hsic_eps_*_async.csv`, `experimental-results/adult_race_hsic_eps_*_sync.csv` |
| Figure 3 | `experimental-results/mnist_noise_*.csv` |
| Figure 4 | `experimental-results/adult_race_hsic_interval_length_*_async.csv`, `experimental-results/adult_race_hsic_interval_length_*_sync.csv` |

### 3. Generate the image PDFs in `experimental-results/images/`

After the CSVs exist, run the plotting functions in `plot_results.py`.

```bash
# Figure 1 images
python -c "import plot_results as p; p.plot_powertrace_decrease_eps(); p.plot_powertrace_boxplots_eps(); p.plot_powertrace_decrease_interval_length(); p.plot_powertrace_boxplots_interval_length()"

# Figure 2 images
python -c "import plot_results as p; p.plot_adult_decrease_eps(); p.plot_adult_decrease_eps_boxplot()"

# Figure 3 images
python -c "import plot_results as p; p.plot_mnist_quantiles(); p.plot_mnist_execution_active_monitors(); p.plot_mnist_execution_values()"

# Figure 4 images
python -c "import plot_results as p; p.plot_adult_decrease_interval_length(); p.plot_adult_decrease_interval_length_boxplot()"
```

The generated image files used in Figures 1-4 are:

| Figure | Image PDFs |
| --- | --- |
| Figure 1 | `powertrace_decrease_eps.pdf`, `powertrace_decrease_eps_boxplots.pdf`, `powertrace_decrease_target_length.pdf`, `powertrace_decrease_target_length_boxplots.pdf` |
| Figure 2 | `adult_decrease_eps.pdf`, `adult_decrease_eps_boxplot.pdf` |
| Figure 3 | `mnist_increase_noise.pdf`, `mnst_active_monitors_execution.pdf`, `mnst_accuracy_values_execution.pdf` |
| Figure 4 | `adult_decrease_interval_lenght.pdf`, `adult_decrease_interval_boxplot.pdf` |


## Reproducing the Stochastic Monitoring Experiments in the Appendix

The standalone script `reproduce_stochastic_monitoring.py` reproduces the synthetic Beta
process experiments from Appendix F.3-F.4, including Figures 5-12 and the Monte
Carlo table.

The script uses only synthetic data, so no external datasets or trained models
are required. The script is documented with function-level docstrings. It uses
the same Python environment as the main artifact workflow;
make sure `scipy`, `numpy`, `pandas`, `matplotlib`, and `seaborn` are installed.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install numpy pandas matplotlib scipy seaborn
```

Run the full stochastic workflow with:

```bash
python reproduce_stochastic_monitoring.py
```

By default this writes all result data, plots, and tables to:

```text
experimental-results/stochastic-monitoring/
```

The paper table uses 1000 Monte Carlo repetitions. The displayed one-run
simulation uses the appendix SIM Beta parameters `(a,b)`. The Monte Carlo
table reproduces both paper blocks by default: first the high-variance
`(a,b)` block and then the lower-variance `(10a,10b)` block. To run only the
lower-variance block, add `--mc-only-low-variance`.

To run a fast smoke test before the full reproduction, use:

```bash
python reproduce_stochastic_monitoring.py --quick
```

To regenerate only the deterministic/single-run plots and skip the Monte Carlo
table, use:

```bash
python reproduce_stochastic_monitoring.py --skip-mc
```

To skip the Appendix F.4 width experiments for Figures 10-12, use:

```bash
python reproduce_stochastic_monitoring.py --skip-rq4
```

Other useful options are `--seed <N>`, `--sim-probe-stride <K>`,
`--no-csv`, and `--n-mc <N>`.

The main outputs are:

| Paper item | Output file |
| --- | --- |
| Figure 5 | `figure_05_beta_process.pdf` |
| Figure 6 | `figure_06_verdict_heatmap.pdf` |
| Figure 7 | `figure_07_uncertainty_evolution.pdf` |
| Figure 8 | `figure_08_width_decomposition.pdf` |
| Figure 9 | `figure_09a_ci_at_first_verdict.pdf`, `figure_09b_ci_at_end.pdf` |
| Figure 10 | `figure_10_convergence.pdf` |
| Figure 11 | `figure_11_limit_discount.pdf` |
| Figure 12 | `figure_12_uniform_time.pdf`, `figure_12_uniform_time_specific.pdf` |
| Table 2 | `table_02_mc_metrics.tex`, `figure_table_02_mc_violation_rates.pdf`, plus CSV summaries |

The script also writes reusable CSV data, including `stochastic_beta_process.csv`,
`stochastic_monitor_grid.csv`, `stochastic_monitor_first_verdicts.csv`,
`table_02_mc_metrics_raw.csv`, `table_02_mc_metrics_summary.csv`, and the data
underlying Figures 10-12. Use `--output-dir <path>` to place the outputs
elsewhere, and `--n-mc <N>` to change the number of Monte Carlo repetitions.

## Repository Structure

`main.py`
: Implements the discounted-sum monitor, the discounted demographic-parity
monitor, and the experiment runners that produce CSV files in
`experimental-results/`.

`plot_results.py`
: Reads the result CSVs and writes the PDF plots in
`experimental-results/images/`.

`experimental-setups/`
: JSON parameter files for each experiment. Each entry specifies the source
trace, monitored column, discount factors, tolerance, target interval, and
output CSV path.

`experimental-results/`
: Generated experiment outputs. The top-level CSV files are monitor runs,
`images/` contains the paper plots, and `MLmodels/` contains local trained
models used to regenerate fairness and MNIST traces when present.

`google-trace-data/`
: Source power trace data used by the Figure 1 experiments.

`ffb_datasets/`, `ffb/`, `ffb_traces/`
: Fair Fairness Benchmark data, model-training code, and generated event
traces. The paper figures use the Adult race HSIC trace
`ffb_traces/adult_race_hsic.csv`.

`mnist-traces/`, `data/`
: Generated MNIST drift traces and the raw MNIST data used to generate them.

`generate_experimental_setups.py`
: Recreates the JSON setup files for the power-trace, Adult, and stochastic
experiments.

`generate_ffb_traces.py`
: Generates the Adult event trace used by the discounted demographic-parity
monitor.

`train_mnist.py`, `generate_mnist_traces.py`
: Train the MNIST CNN and generate MNIST drift traces with increasing noise.

`generate_stochastic_traces.py`
: Generates synthetic stochastic traces. These are not needed for Figures 1-4.

`reproduce_stochastic_monitoring.py`
: Reproduces the stochastic discounted-sum monitoring experiments from the
synthetic Beta process, including Figures 5-12, Table 2, and the corresponding
CSV data in `experimental-results/stochastic-monitoring/`.
