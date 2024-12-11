# Use the official Miniconda3 image as the base
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yaml into the container
COPY environment.yaml .

# Create the conda environment
RUN conda env create -f environment.yaml

# Activate the environment and ensure it's activated
RUN echo "conda activate mrcnn_env" >> ~/.bashrc

# Copy the rest of your application code into the container
COPY . .

# Set the entry point to run your application within the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mrcnn_env", "python", "mask_rcnn/inference/food_predict_flask.py"]
