## Solutions for the kaggle competition tabular playground


### Git
```
git clone https://github.com/wwagner4/tabplay-py.git
```


### Run in docker

```
docker run \
-v /home/itsv.org.sv-services.at/31100428/prj/tabplay-py:/opt/project \
-v /home/itsv.org.sv-services.at/31100428/work1/kaggle/tabplay:/opt/work \
--rm \
tabplay \
python -u /opt/project/tabplay/tryout.py 01
```

```
# local submission
docker run \
--detach \
-v /home/itsv.org.sv-services.at/31100428/prj/tabplay-py:/opt/project \
-v /home/itsv.org.sv-services.at/31100428/work1/kaggle/tabplay:/opt/work \
--rm \
tabplay \
python -u /opt/project/tabplay/localsubm.py

# gradient boost
docker run \
--detach \
-v /home/itsv.org.sv-services.at/31100428/prj/tabplay-py:/opt/project \
-v /home/itsv.org.sv-services.at/31100428/work1/kaggle/tabplay:/opt/work \
--rm \
tabplay \
python -u /opt/project/tabplay/gbm.py 02

# random forest @ work
docker run \
--detach \
-v /home/itsv.org.sv-services.at/31100428/prj/tabplay-py:/opt/project \
-v /home/itsv.org.sv-services.at/31100428/work1/kaggle/tabplay:/opt/work \
--rm \
tabplay \
python -u /opt/project/tabplay/random_forest.py 01

# split target @ ben
docker run \
--detach \
-v /home/wwagner4/prj/oldschool/tabplay-py:/opt/project \
-v /data/work/tabplay:/opt/work \
--rm \
tabplay \
python -u /opt/project/tabplay/splittarget.py

```
