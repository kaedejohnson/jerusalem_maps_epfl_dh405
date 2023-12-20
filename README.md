# jerusalem_maps_epfl_dh405

## Setup:
Installing the mapKurator spotter on Windows is rather cumbersome due to precise package requirements and source code alterations not listed in the mapKurator documentation. Those wishing to recreate and investigate our novel pipeline may wish to forgo installation of mapKurator's spotter and instead use our results from mapKurator's unaltered spotter module; this can be done by beginning at "Processing" below.

### Installation
#### Windows:
1. Get your mapKurator virtual environment setup:
- Instructions [here](https://knowledge-computing.github.io/mapkurator-doc/#/docs/install1). Do everything until "Download OpenStreetMap data..." except PostgreSQL and elasticsearch.
- Detectron2 must be version 0.6 !

2. Download model weights [here](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view). You'll specify your local location in #5.

3. Use inference.py from this repo's dependencies folder to overwrite inference.py from your local mapkurator-spotter-main\spotter-v2\tools\ git clone.

4. Download the latest raw_maps folder and pipeline.ipynb from this repo.

5. Replace filepaths in the second code cell of pipeline.ipynb if necessary.

#### Linux:
1. Get your mapKurator virtual environment setup:
- Instructions [here](https://knowledge-computing.github.io/mapkurator-doc/#/docs/install1). Do everything until "Download OpenStreetMap data..." except PostgreSQL and elasticsearch.
- Detectron2 must be version 0.6 !

2. Download model weights [here](https://drive.google.com/file/d/1agOzYbhZPDVR-nqRc31_S6xu8yR5G1KQ/view). You'll specify your local location in #5.

3. Skip step 3   

4. Download the latest raw_maps folder and pipeline.ipynb from this repo.

5. Replace filepaths in the second code cell of pipeline.ipynb if necessary.

### Processing
You must setup correct paths in step #5 above. To run our pipeline only (not the mapKurator spotter), run cells from "3: Label combination" onward. This processes one map at a time, as specified by map_name_in_strec.

### Evaluation
When you have the results from three maps (processed/strec/{map_name_in_strec}/fully_processed_labels.pickle), run cells from "4: Evaluation". Recalls and precisions for each map will be printed.

## Results:
Please refer to our [Wiki page](https://fdh.epfl.ch/index.php/Extracting_Toponyms_from_Maps_of_Jerusalem) for detailed methods and analysis.
