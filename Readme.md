## Solutions for the kaggle competition tabular playground

### Run in docker

```
docker run \
-v /home/itsv.org.sv-services.at/31100428/prj/tabplay-py:/opt/project \
-v /home/itsv.org.sv-services.at/31100428/work1/kaggle/tabplay:/opt/work \
--rm \
tabplay \
python /opt/project/tabplay/tryout.py 01
```

```
docker run \
--detach \
-v /home/itsv.org.sv-services.at/31100428/prj/tabplay-py:/opt/project \
-v /home/itsv.org.sv-services.at/31100428/work1/kaggle/tabplay:/opt/work \
--rm \
tabplay \
python /opt/project/tabplay/gbm.py 02
```