# Cellpose Pre-trained Models

## sep_model_1180
This model was trained using 8 images with a DIC and a DAPI channel and 1180 ROIs in the Cellpose GUI. 

The parameters used to train the model are as follows:
initial model: cyto
chan to segment: red
cha2 (optional): green
learning_rate: 0.1
weight_decay: 0.0001
n_epochs: 100

This model was trained to recognise buds separately to the mother cells.

## whole_model_993
This model was trained using 8 images with a DIC and a DAPI channel and 993 ROIs in the Cellpose GUI. 

The parameters used to train the model are as follows:
initial model: cyto
chan to segment: red
cha2 (optional): green
learning_rate: 0.1
weight_decay: 0.0001
n_epochs: 100

This model was trained to recognise buds and mother fragments as one cell.

## shifted_whole_model_993
This model was trained using 8 images with a DIC and a **shift corrected** DAPI channel and 993 ROIs in the Cellpose GUI. 

The parameters used to train the model are as follows:
initial model: cyto
chan to segment: red
cha2 (optional): green
learning_rate: 0.1
weight_decay: 0.0001
n_epochs: 100

This model was trained to recognise buds and mother fragments as one cell.
