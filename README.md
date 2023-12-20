# jerusalem_maps_epfl_dh405

## Setup:

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
You must setup correct paths in step #5. Steps from "3: Label combination" on, process one map at a time, specify the map name with varialble map_name_in_strec.

### Evaluation
When you have the results from three maps, run evaluation section. Recalls and precisions for each methods will be printed.

## Results:
Please refer to our [Wiki page](https://fdh.epfl.ch/index.php/Extracting_Toponyms_from_Maps_of_Jerusalem) for detailed results and analysis.
