mkdir -p src/data/input/kitti
mkdir -p src/data/output/kitti

wget "https://www.dropbox.com/scl/fi/7foz4er9snxed7equq1yo/skitti.zip?rlkey=zz687lyi9b0vju3byl0jnnqub&dl=1" -0 src/data/input/kitti/kitti.zip

unzip src/data/input/kitti/kitti.zip -d src/data/input/kitti

rm src/data/input/kitti/kitti.zip