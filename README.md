# jerusalem_maps_epfl_dh405

SETUP:

Get your mapKurator virtual environment setup:
    Do everything until "Download OpenStreetMap data and create indices for PostOCR and Entity Linker modules" [here](https://knowledge-computing.github.io/mapkurator-doc/#/docs/install1), except PostgreSQL and elasticsearch.
    ! Detectron2 must be version 0.6 !

Download model weights [here](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view)

Overwrite inference.py from your local mapkurator-spotter-main\spotter-v2\tools\ git clone with inference.py from this repository's dependencies folder

Download the latest raw_maps folder and pipeline.ipynb

Replace filepaths in the second code cell of pipeline.ipynb if necessary

