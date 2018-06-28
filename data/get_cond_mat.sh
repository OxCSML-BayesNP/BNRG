wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \
    -O cond_mat.txt \
    http://opsahl.co.uk/tnet/datasets/Newman-Cond_mat_95-99-Newman.txt
python parse.py cond_mat.txt
