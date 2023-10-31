# jerusalem_maps_epfl_dh405

SETUP:

1. Get your mapKurator virtual environment setup:
- Instructions [here](https://knowledge-computing.github.io/mapkurator-doc/#/docs/install1). Do everything until "Download OpenStreetMap data..." except PostgreSQL and elasticsearch.
- Detectron2 must be version 0.6 !

2. Download model weights [here](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view). You'll specify your local location in #5.

3. Use inference.py from this repo's dependencies folder to overwrite inference.py from your local mapkurator-spotter-main\spotter-v2\tools\ git clone.

4. Download the latest raw_maps folder and pipeline.ipynb from this repo.

5. Replace filepaths in the second code cell of pipeline.ipynb if necessary.

