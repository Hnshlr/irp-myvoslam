mkdir -p src/data/input/kitti
mkdir -p src/data/output/kitti

wget "https://www.dropbox.com/scl/fi/moukxxg7eeifv45w259e6/skitti.zip?rlkey=2xpoionnbai54vepbql8xm490&dl=1" -0 src/data/input/kitti/skitti.zip

unzip src/data/input/kitti/skitti.zip -d src/data/input/kitti

rm src/data/input/kitti/skitti.zip

mkdir -p src/models

wget "https://www.dropbox.com/scl/fi/geevv3l7vv175zor79vjn/models.zip?rlkey=a9k4mfn1dl9lv371lhdvjmnn7&dl=1" -0 src/models.zip

unzip src/models.zip -d src/models

rm src/models.zip