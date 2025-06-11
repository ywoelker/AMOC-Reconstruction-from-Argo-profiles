# Estimating AMOC from Argo Profiles with Machine Learning Trained on Ocean Simulations

This project is the additional code to our manuscript "Estimating AMOC from Argo Profiles with Machine Learning Trained on Ocean Simulations" with all information how to train the AMOC reconstruction and jupyter notebook to generate the figures.

### Abstract 
The Atlantic Meridional Overturning Circulation (AMOC) plays an important role in our climate system, continuous monitoring is important and could be enhanced by combing all available information.
Moored measuring arrays like RAPID divide the AMOC in near-surface contributions, western-boundary currents, and the deep ocean in the interior of the basin.
For the deep-ocean component, moorings measure density and focus on the calculation through geostrophy. These moored devices come with a high maintenance effort. 
Existing reconstruction studies show success with near-surface variables on monthly time scales, but do not focus on the interior transport. 
For interannual to decadal time scales, the geostrophic contribution becomes an important contribution.

Argo floats could provide required information about the geostrophic circulation as they continuously and cost-effective deliver hydrographic profiles. But they are spatially unstructured and only report instantaneous values.
Here we show that the geostrophic part of the AMOC can be data-drivenly reconstructed by Argo profiles. To demonstrate this, we use a realistic and physically consistent high-resolution model VIKING20X.
By simulating virtual Argo floats, we demonstrate that a learnable binning method to process the spatially variable Argo float distribution is able to reconstruct the geostrophic part of the VIKING20X AMOC by up to 80\% explained variance and a mean error of less than one Sverdrup for the geostrophic transport.
For this reconstruction, we sampled virtual Argo profiles mimicking their historical distribution in the real ocean. This enables us to apply the trained reconstruction to real-world RAPID data, showing large potential for a generalization into the real world.
Our results demonstrate how an AMOC reconstruction from unstructured Argo profiles could replace estimates of the geostrophic deep-ocean component of the AMOC from the RAPID Array in the context of high-resolution ocean and climate models.



## 1. Step downloading the dataset

The AMOC reconstruction is trained on virtual moorings and Argo profiles which are extracted from the high-resolution ocean model VIKING20X. All input data are published in a zenodo data publication and can be reached at this doi. The first step is to download the dataset and extract it. The dafault path of the projects reference the dataset publication on the same folder level as this repository.

```
|
|-data_publication
|  |- datasets
|  `- introduction_figure_support
|-paper_code_amoc_reconstruction
...
```


## 2. Step Train a reconstruction (optional)

Create a environment with the `requirements.txt` from the repository to have all dependencies. Then install the base repository that holds all the preprocessing code into the environment. (see Install in [amoc_reconstruction]())

As the dataset from step one is not complete the transport timeseries have to be computed before the reconstruction can be executed.

```bash
sh scripts/create_transport_time_series.sh
```

If the dataset is located at a different location than in Step 1 you have to adjust the base_path in the script.


Now the reconstruction can be trained with the `scripts/execute_experiment.py` script. The help page should be helpful an example is:

```shell
python scripts/execute_experiment.py --input_smoothing 90 --input_cycles 1,2 --test_period 2004,2024
```


To reporduce all experiments that we used for the study you can execute the shell-script `scripts/paperdraft_experiments.job` which is written for the local HPC center but can easily be executed as a shell script. 

```bash 
sh scripts/paperdraft_experiments.job
```

The script will create a `experiments/paperdraft.json` file which keeps track of the expriments and has an id per experiment. The experiments will be saved in `experiments/results`. The experiment identifier are important for the figure creation.

## 3. Step Download paper results 
We published in a seperate zenodo repository also the experiment output data that was used for the origianal paper. If you are interested in reproducing only the figures or do not have time/resources to reproduce all experiments, you can download the original content of the experiments folder. 
The zenodo doi is [doi.org]() and the dataset can be downloaded and extracted to the experiments folder. This is an example structure of the experiment folder: 

```txt
|- experiments 
|   |- results 
|   |   |- experiment_20250328_141208_abcd
|   |   |   |- member_xy
|   |   |   |   |- skill_images.png
|   |   |   |   |- model_xy.pt
|   |   |   |   |- test_predictions.nc
|   |   |   |   `- metric.json
|   |   |  ...
|   |   |   |- configuration.json
|   |   |   |- ensemble_metrics.json
|   |   |   `- ensemble_predictions.nc
|   |  ...
|   ` paperdraft.json

```

## 4. Step Recreate the figures of the paper
To recreate the figures of the study use the juypter notebooks in the `notebooks`folder. The first notebook `14_figures_paper.ipynb` contains the introduction figure as well as some smaller icons that are part of Figure 2. All other notebooks are part of the result section with the following mapping of sections to notebooks: 

```txt
Sec 4.1: 25_paperdraft_fig_performance.ipynb
Sec 4.1.1: 29_paperdraft_allcycles.ipynb
Sec 4.2: 26_paperdraft_fig_importance.ipynb
Sec 4.3: 27_paperdraft_fig_geostrophic.ipynb
Sec 4.4: 28_paperdraft_fig_deep_information.ipynb
```

