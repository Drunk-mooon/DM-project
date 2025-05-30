# most naive
# output: "node_id" : [list of most probable neighbor_id and similarity]
python similarity.py --emb embeddings.npy --topk 5 --output topk_neighbors.json --metric cosine
python similarity.py --emb embeddings.npy --topk 5 --output topk_neighbors.json --metric jaccard
python similarity.py --emb embeddings.npy --topk 5 --output topk_neighbors.json --metric euclidean

# build a VAE and use its representation instead of raw features
python train.py --out_dir ./ --data_path ./embeddings.npy --batch_size 1024 # note to alter the batch size
python transform_emb.py --data_path embeddings.npy --checkpoint_path vae_checkpoint.pt --output_path transformed_emb.npy
python similarity.py --emb transformed_emb.npy --topk 5 --output topk_neighbors.json --metric euclidean

# DEC
python dec.py --emb embeddings.npy -o dec_output # here we have a cluster result stored in y_pred.npy
python similarity.py --emb dec_output/latent.npy --topk 5 --output topk_neighbors.json --metric euclidean # predict with latent given by DEC


# run ALL graph based method
# output: "node_id" : cluster_id
python graph_cluster.py --emb embeddings.npy -o graph_out --k 25
python graph_cluster.py --emb transformed_emb.npy -o graph_out --k 25