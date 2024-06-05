# DREW: Data Retrieval with Error-Corrected Codes and Watermarking

## Running the code

The script `_bash_full_run.sh $DS_NAME $DATA_DIR` can be executed by inputting a name for the dataset, and the path to the images in the dataset. This script will in order:

- Apply the watermark to the images in the dataset and save the watermarked images, together with the info of the clusters, cluster codes, and watermark keys in a folder.
- Create latent embeddings for the images in the dataset and and store them in a file.
- Sample the queries and store them for a list of augmention types with different severity levels.
- Evaluate the retrieval accuracy for our method and the naive retrieval, and store the logs in a file.


There are some other parameters of the method that can be changed in the bash script, such as the watermarking method $\mathcal{W}$, the embedding model $\phi$, number of watermark bits $n$, and number of clusters $2^k$.