wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz
gzip -d com-dblp.ungraph.txt.gz
gzip -d com-dblp.top5000.cmty.txt.gz
sed '1,4d' email-Enron.txt > enron.txt
python make_dblp.py
