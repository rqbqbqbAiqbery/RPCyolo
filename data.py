import kagglehub

# Download latest version
path = kagglehub.dataset_download("diyer22/retail-product-checkout-dataset")

print("Path to dataset files:", path)