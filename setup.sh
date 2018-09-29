## download cond-mat
#wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \
#    -O data/cond_mat.txt \
#    http://opsahl.co.uk/tnet/datasets/Newman-Cond_mat_95-99-Newman.txt
## make cond-mat
#python -m data.make_cond_mat
#
## download enron
#wget https://snap.stanford.edu/data/email-Enron.txt.gz -P data
#gzip -d data/email-Enron.txt.gz
#sed '1,4d' data/email-Enron.txt > data/enron.txt
## make enron
#python -m data.make_enron
#
## download internet
#wget https://www.cise.ufl.edu/research/sparse/mat/Pajek/internet.mat -P data
## make internet
#python -m data.make_internet
#
## download polblogs
#wget https://www.cise.ufl.edu/research/sparse/mat/Newman/polblogs.mat -P data
## make polblogs
#python -m data.make_polblogs
#
## download dblp
#wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz -P data
#wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz -P data
#gzip -d data/com-dblp.ungraph.txt.gz
#gzip -d data/com-dblp.top5000.cmty.txt.gz
## make dblp
#python -m data.make_dblp

rm data/*.txt
rm data/*.mat

# compile gigrnd
cd utils
make
cd ..
