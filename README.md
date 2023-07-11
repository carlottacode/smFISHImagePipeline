# smFISHimagepipeline
### Environment 
conda create -n carlotta_env python=3.9 -y
conda activate carlotta_env
pip install napari[all]
pip install big-fish==0.6.2
pip install stackview==0.6.3
pip install scikit-image==0.21.0
conda install -c anaconda opencv==4.6.0
conda install -c conda-forge jupyterlab==3.5.0
pip install shapely==2.0.1
python -m pip install cellpose[gui]
jupyter-notebook

