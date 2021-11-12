apt-get install -y -q g++
apt-get update
apt-get install -y -q openjdk-8-jdk python-dev python3-dev
yes | pip install konlpy jpype1-py3

# install khaiii
cd ~/tmp
git clone https://github.com/kakao/khaiii.git
cd khaiii
mkdir build
cd build
yes | pip install -q cmake
apt-get install -y -q cmake
cmake ..
make resource
make install
make package_python
cd package_python
yes | pip install -q .
cd ~/tmp
apt-get install -y -q locales
locale-gen en_US.UTF-8
yes | pip install -q tweepy==3.7.0

# install mecab
cd ~/tmp
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
rm -f mecab-0.996-ko-0.9.2.tar.gz

wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
rm -f mecab-ko-dic-2.1.1-20180720.tar.gz

cd mecab-ko-dic-2.1.1-20180720
ldconfig

cd ~/tmp/mecab-0.996-ko-0.9.2
./configure
make
make check
make install

cd ~/tmpmecab-ko-dic-2.1.1-20180720
./autogen.sh
./configure
make
make install

cd ~/tmp
mecab -d /usr/local/lib/mecab/dic/mecab-ko-dic
apt-get install -y -q curl git
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
yes | pip install -q mecab-python