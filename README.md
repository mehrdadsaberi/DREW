# DREW: Data Retrieval with Error-Corrected Codes and Watermarking

<!-- 
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.02836)

> **DREW : Towards Robust Data Provenance by Leveraging Error-Controlled Watermarking**<br>
> [Mehrdad Saberi](https://github.com/mehrdadsaberi), [Vinu Sankar Sadasivan](), [Arman Zarei](), [Hessam Mahdavifar](), [Soheil Feizi]()<br> -->



## Setup

Create a conda environment and install required packages:
```
conda create drew_env python=3.11
conda activate drew_env
pip install -r requirements.txt
```


To use Diffusion Purification augmentation, download the model using the following commands:
```
mkdir -p ./DiffPure/pretrained/guided_diffusion
wget -O ./DiffPure/pretrained/guided_diffusion/256x256_diffusion_uncond.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

## Running the code

The script `_bash_full_run.sh $DS_NAME $DATA_DIR` can be executed by inputting a name for the dataset, and the path to the images in the dataset. This script will in order:

- Apply the watermark to the images in the dataset and save the watermarked images, together with the info of the clusters, cluster codes, and watermark keys in a folder.
- Create latent embeddings for the images in the dataset and and store them in a file.
- Sample the queries and store them for a list of augmention types with different severity levels.
- Evaluate the retrieval accuracy for our method and the naive retrieval, and store the logs in a file.


There are some other parameters of the method that can be changed in the bash script, such as the watermarking method $\mathcal{W}$, the embedding model $\phi$, number of watermark bits $n$, and number of clusters $2^k$.


We note that some parts of the current code are not GPU optimized and they might be updated in the future.