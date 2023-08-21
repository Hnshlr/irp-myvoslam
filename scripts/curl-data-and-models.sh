mkdir -p src/data/input/kitti
mkdir -p src/data/output/kitti

curl -L "https://www.dropbox.com/scl/fi/moukxxg7eeifv45w259e6/skitti.zip?rlkey=2xpoionnbai54vepbql8xm490&dl=1" -o src/data/input/kitti/skitti.zip

unzip src/data/input/kitti/skitti.zip -d src/data/input/kitti

rm src/data/input/kitti/skitti.zip

mkdir -p src/models

curl -L "https://www.dropbox.com/scl/fi/vkg6h6pwvad67jptc790q/models.zip?rlkey=w67evmh3mfwtxdvbuq8xsuq95&dl=1" -o src/models/models.zip

unzip src/models/models.zip -d src/models

rm src/models/models.zip