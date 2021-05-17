# Face Detection

Create a face detection model from scratch on Grid. This project covers:

1. creating a dataset from sratch
2. creating a [Grid AI](https://grid.ai) Datastore
3. training a model on [Grid AI](https://grid.ai)
4. using your trained model for inference

## Step 1: Create Your Dataset

We will be creating a dataset with the following folder structure in order to use this project:

```shell
dataset
├── test
│   ├── face_a
│   │   ├── 528bca97-9676-4d4b-866e-67ace4469ffc_output.jpg
│   │   └── e0facb84-beee-4210-a09f-9fd60dd0bc6a_output.jpg
│   └── face_b
│       ├── 0164e6da-f2d3-423c-a3fc-e3469993fb7b_output.jpg
│       └── IMG_8021_output.jpg
├── test
│   ├── face_a
│   │   └── e0facb84-beee-4210-a09f-9fd60dd0bc6a_output.jpg
│   ├── face_b
│   │   ├── 0164e6da-f2d3-423c-a3fc-e3469993fb7b_output.jpg
│   │   └── IMG_8021_output.jpg
└── val
    ├── face_a
    │   └── e0facb84-beee-4210-a09f-9fd60dd0bc6a_output.jpg
    └── face_b
        ├── 0164e6da-f2d3-423c-a3fc-e3469993fb7b_output.jpg
        └── IMG_8021_output.jpg
```

Each face you want to detect corresponds to a directory name. And each root directory corresponds to a
different split of the dataset, i.e. `train`, `test`, and `val`.

### Step 1.1: Process Raw Data

Place all your image files in the the `raw` directory. Don't organize them into folders yet, just place them in the root
of that directory.

We will not be running those images through a face-detection model called [MTCNN](https://arxiv.org/pdf/1604.02878.pdf).
We'll then be cropping all the detected faces into their own files and storing the output in the `./processed` directory.
Let's do that with:

```shell
$ python process_raw_data.py
```

You will find a number of really small images with only faces. You now need to "annotate" the resulting images by doing
two things:

1. removing any images that aren't faces (MTCNN makes mistakes sometimes)
2. placing each image in a directory with the same of the person whose face belongs to, for example:

```shell
# `vera` and `luis` are the two people in this dataset
processed/vera/photo_1.jpg
processed/luis/photo_1.jpg
```

### Step 1.2: Create Dataset Splits

We will now be splitting your "annotated" dataset into three collections: `train`, `test`, and `val`. You can
do that by running the script: `create_training_dataset.py`

```shell
$ python create_training_dataset.py
```

You will now find a new directory called `./dataset`. This directory contains the training dataset you need.

This script also generates a file that maps label indices to class names. That file is called `dataset/labels.json` and has the following format:

```json
{"labels": ["label_a", "label_b"]}
```

We'll be using that file later when using the trained model for predictions.

## Step 2: Train Your Model

[Grid AI](https://grid.ai) introduces the concept of [Datastores](https://docs.grid.ai/products/add-data-to-grid-datastores). Datastores are
high performance volumes mounted into your training context when using Grid. That means that you can create a Datastore once and then use
it to create both [Sessions](https://docs.grid.ai/products/sessions) or [Runs](https://docs.grid.ai/products/run-run-and-sweep-github-files) on Grid.

Make sure to install the [Grid CLI and login](https://docs.grid.ai/products/global-cli-configs), then:

```shell
$ grid datastores create --name face_detection --source dataset/
upload ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0%
✔ Finished uploading datastore.
```

You can then verify that your datastore is ready to use by checking its status:

```shell
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Credential Id ┃           Name ┃ Version ┃     Size ┃          Created ┃    Status ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cc-bwhth      │ face_detection │       1 │   2.1 MB │ 2021-05-13 19:55 │ Succeeded │
└───────────────┴────────────────┴─────────┴──────────┴──────────────────┴───────────┘
```

Whenever your datastore has status `Succeeded` it is ready to use.

## Step 3: Train Your Model

You can train your model by calling the `train.py` script locally (make sure to install your dependencies first):

```shell
$ python3.8 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
$ python train.py
Global seed set to 1234
train samples: 341
valid samples: 42
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

  | Name     | Type       | Params
----------------------------------------
0 | metrics  | ModuleDict | 0     
1 | backbone | Sequential | 11.2 M
2 | head     | Sequential | 1.5 K 
----------------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.712    Total estimated model params size (MB)
Validation sanity check:   0%|                     | 0/2 [00:00<?, ?it/s]
```

Feel free to run that locally to test that your model works as expected. 

### Step 3.1: Train Your Model on Grid AI

You are now ready to train your model on Grid. We'll be using the CLI but you can do the same thing by using the
web UI. We have placed a configuration file locally (`.grid/config.yml`) that you can use as reference instead of
passing all the parameters to the CLI manually -- or just click on Grid badge:

[![Grid](https://img.shields.io/badge/rid_AI-run-78FF96.svg?labelColor=black&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMTR2MjBhMTQgMTQgMCAwMDE0IDE0aDlWMzYuOEgxMi42VjExaDIyLjV2N2gxMS4yVjE0QTE0IDE0IDAgMDAzMi40IDBIMTVBMTQgMTQgMCAwMDEgMTR6IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTM1LjIgNDhoMTEuMlYyNS41SDIzLjl2MTEuM2gxMS4zVjQ4eiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://platform.grid.ai/#/runs?script=https://github.com/luiscape/gridai-face-recognition/blob/857fc2268a49a55b1f5c8412f3b43c616dd4a33f/train.py&cloud=grid&instance=g4dn.xlarge&accelerators=1&disk_size=200&framework=lightning&script_args=grid%20run%20--grid_instance_type%20g4dn.xlarge%20--grid_gpus%201%20--grid_datastore_name%20face_detection%20--grid_datastore_version%201%20--grid_use_spot%20--grid_datastore_mount_dir%20%2Fgridai%2Fproject%2Fdataset%20train.py%20--max_epochs%201000%20--data_path%20%2Fgridai%2Fproject%2Fdataset)

```shell
$ grid run --grid_instance_type g4dn.xlarge \
           --grid_gpus 1 \
           --grid_datastore_name face_detection \
           --grid_datastore_version 1 \
           --grid_datastore_mount_dir /gridai/project/dataset \
           train.py --max_epochs 1000 --data_path /gridai/project/dataset

No --grid_name passed, naming your run glossy-manatee-255
Using default cloud credentials cc-bwhth to run on AWS.

                Run submitted!
                `grid status` to list all runs
                `grid status glossy-manatee-255` to see all experiments for this run

                ----------------------
                Submission summary
                ----------------------
                script:                  train.py
                instance_type:           g4dn.xlarge
                distributed:             False
                use_spot:                True
                cloud_provider:          aws
                cloud_credentials:       cc-bwhth
                grid_name:               glossy-manatee-255
                datastore_name:          face_detection
                datastore_version:       1
                datastore_mount_dir:     /gridai/project/dataset
```

### Bonus: Run a Hyperparameter Sweep

Grid AI makes it trivial to run a [hyperparameter sweep](https://docs.grid.ai/products/global-cli-configs/cli-api/grid-train#hyperparameter-sweeps)
without having to change anything in your scripts. The model we created provides support for a number of different backbones,
including `resnet18` and `resnet200d`. Let's try both different models and learning rates to make sure we find the best model:

```shell
$ grid run --grid_instance_type g4dn.xlarge \
           --grid_gpus 1 \
           --grid_datastore_name face_detection \
           --grid_datastore_version 1 \
           --grid_datastore_mount_dir /gridai/project/dataset \
           train.py --max_epochs 1000 --data_path /gridai/project/dataset \
                    --learning_rate "uniform(0,0.0001,2)" --backbone "['resnet18','resnet200d']"
```

That will generate 4 experiments combining both different backbones and learning rate combinations.

## Step 4: Predict

This section covers how to get your weights from Grid and make predictions with your model.

### Step 4.1: Get Your Weights

Let's download your latest weights from Grid and run a series of predictions with your trained model.
We'll first download all artifacts from your run with `grid artifacts`. In this case my Run was called
`glossy-manatee-255`. When I run `grid artifacts glossy-manatee-255` it downloads all the artifacts for
the Experiments from that Run.

```shell
$ grid artifacts glossy-manatee-255

Downloading artifacts → ('glossy-manatee-255',)
  glossy-manatee-255 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

Artifacts are saved by default in the `grid_artifacts` directory:

```shell
$ tree grid_artifacts
grid_artifacts
└── glossy-manatee-255
    └── glossy-manatee-255-exp0
        └── version_0
            ├── checkpoints
            │   └── epoch=712-step=7129.ckpt
            ├── events.out.tfevents.1620938447.exp-glossy-manatee-255-exp0.20.0
            └── hparams.yaml

4 directories, 3 files
```

The file we are looking for is `epoch=712-step=7129.ckpt` which is the latest PyTorch checkpoint file.

### Step 4.2: Load Your Weights And Make Predictions

Now that we have our weights locally we want to load them using [Lightning Flash](https://github.com/PyTorchLightning/lightning-flash) and make predictions. You can run the script `predict.py` to test your new trained model:

```shell
$ python predict.py --checkpoint_path grid_artifacts/glossy-manatee-255/glossy-manatee-255-exp0/version_0/checkpoints/epoch=712-step=7129.ckpt \
                    --image_path test_prediction_image.jpg
Predicted class: person_a
```
