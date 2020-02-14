* install prerequisites, build and test as a pipeline
```shell
# sudo privilege is needed
# please excute the command at the project root path
# BMNNSDK_PATH is the absolute path to install BMNNSDK
cd ${BMNNSDK_PATH};source envsetup.sh
bash ./tools/test_all.sh 
```
* install prerequisites
```shell
# sudo privilege is needed
bash ./tools/install.sh
```
* downlad test data, model and convert to bmodel
```shell
# get the model list
python3 download_and_convert.py model_list
# downlad test data
python3 download_and_convert.py test_data
# downlad a model in model list and convert it to bmodel
python3 download_and_convert.py model_name
# downlad test data, download all models in model list and convert them to bmodels
python3 download_and_convert.py all
```
