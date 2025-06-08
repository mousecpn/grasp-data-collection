# Data Collection for Volumetric Grasping Models

## Installation

1. Create a conda environment.

2. Install packages list in [requirements.txt](requirements.txt).

3. Go to the root directory and install the project locally using `pip`

```
pip install -e .
```

4. Build ConvONets dependents by running `python convonet_setup.py build_ext --inplace`.

5. (Optional) You can install [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) to speed up your data collection.

## Self-supervised Data Generation

### Raw synthetic grasping trials

```bash
./data_collection.sh (giga | graspnet | contact) (pile | packed) /path/to/raw/data num_grasps
```

### Data clean and processing

First clean and balance the data using:

```bash
python scripts/clean_balance_data.py /path/to/raw/data
```

Then construct the dataset (add noise):

```bash
python scripts/construct_dataset_parallel.py --num-proc 40 --single-view --add-noise dex /path/to/raw/data /path/to/new/data
```

### Save occupancy data

Sampling occupancy data on the fly can be very slow and block the training, so I sample and store the occupancy data in files beforehand:

```bash
python scripts/save_occ_data_parallel.py /path/to/raw/data 100000 2 --num-proc 40
```

Please run `python scripts/save_occ_data_parallel.py -h` to print all options.



