mkdir data
cd data

wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
tar -xzf Imagenet.tar.gz

wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
tar -xzf Imagenet_resize.tar.gz

wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xzf LSUN.tar.gz

wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
tar -xzf LSUN_resize.tar.gz

wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
tar -xzf iSUN.tar.gz

rm *.gz

cd ..

mkdir pre_trained
cd pre_trained

wget https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth
wget https://www.dropbox.com/s/7ur9qo81u30od36/densenet_cifar100.pth
wget https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth
wget https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth
wget https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth
wget https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth

cd ..

mkdir output

python OOD_Generate_Mahalanobis.py --dataset cifar100 --net_type resnet --gpu 0
python OOD_Generate_Mahalanobis.py --dataset cifar10 --net_type resnet --gpu 0

python OOD_Regression_Mahalanobis.py --net_type resnet > resnet_results.txt

python OOD_Generate_Mahalanobis.py --dataset cifar10 --batch_size=64 --net_type densenet --gpu 0
python OOD_Generate_Mahalanobis.py --dataset cifar100 --batch_size=64 --net_type densenet --gpu 0

python OOD_Regression_Mahalanobis.py --net_type densenet > densenet_results.txt
