wget https://www.cise.ufl.edu/research/sparse/mat/Newman/polblogs.mat
python process_mat.py polblogs.mat
python parse.py polblogs.txt
