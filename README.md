# jerusalem_maps_epfl_dh405

SETUP:

1. Get your mapKurator virtual environment setup:
-- Do everything until "Download OpenStreetMap data and create indices for PostOCR and Entity Linker modules" [here](https://knowledge-computing.github.io/mapkurator-doc/#/docs/install1), except PostgreSQL and elasticsearch.
-- Detectron2 must be version 0.6 !

2. Download model weights [here](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view)

3. Overwrite inference.py from your local mapkurator-spotter-main\spotter-v2\tools\ git clone with inference.py from this repository's dependencies folder

4. Download the latest raw_maps folder and pipeline.ipynb

5. Replace filepaths in the second code cell of pipeline.ipynb if necessary

