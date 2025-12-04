First go to the dir:
eg. cd denoiser

Build container:
sudo docker build -t denoiser .

Run container:
sudo docker run -it -v "/your/path/of/input_data/:/denoiser/input_data/" -v "/your/path/of/output_data/:/denoiser/output_data" denoiser

eg. /your/path/of/input_data/ - provide the path for your data


Input:
a folder (input_data) with MR volumes (nifti files) 

Output:
denoised examinations (nifti files) with corresponding filenames (output_data)
