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
