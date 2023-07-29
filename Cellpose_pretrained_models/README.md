# Cellpose Pre-trained Models

## sep_model_1180
This model was trained using 8 images with a DIC and a DAPI channel and 1180 ROIs in the Cellpose GUI. 

<p>The parameters used to train the model are as follows:<br>
initial model: cyto<br>
chan to segment: red<br>
cha2 (optional): green<br>
learning_rate: 0.1<br>
weight_decay: 0.0001<br>
n_epochs: 100</p>

This model was trained to recognise buds separately to the mother cells.

## whole_model_993
This model was trained using 8 images with a DIC and a DAPI channel and 993 ROIs in the Cellpose GUI. 

<p>The parameters used to train the model are as follows:<br>
initial model: cyto<br>
chan to segment: red<br>
cha2 (optional): green<br>
learning_rate: 0.1<br>
weight_decay: 0.0001<br>
n_epochs: 100</p>

This model was trained to recognise buds and mother fragments as one cell.

## shifted_whole_model_993
This model was trained using 8 images with a DIC and a **shift corrected** DAPI channel and 993 ROIs in the Cellpose GUI. 

<p>The parameters used to train the model are as follows:<br>
initial model: cyto<br>
chan to segment: red<br>
cha2 (optional): green<br>
learning_rate: 0.1<br>
weight_decay: 0.0001<br>
n_epochs: 100</p>

This model was trained to recognise buds and mother fragments as one cell.
