wget https://snap.stanford.edu/data/email-Enron.txt.gz
gzip -d email-Enron.txt.gz
sed '1,4d' email-Enron.txt > enron.txt
python parse.py enron.txt
