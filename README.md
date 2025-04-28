### Handwriting Generation with Style Transfer
The goal of this project is to train a model that can generate handwriting line path in the same style as example images.

This project aims to combine the style encoder of the One-DM (paper, code) and the ChiroDiff (paper, code) path diffusion model.

## Data
We are using the IAM online handwriting dataset. 

To run this, 
1. Create a src/data directory
2. Download the `original-xml-part.tar.gz`, `lineStrokes-all.tar.gz`, and `lineImages-all.tar.gz`. 
3. Extract the tar files with `tar xf <filename>`

## Timeline
In Progress:
1. Custom Dataloader for IAM dataset.
  I got it mostly working with just a few things I might need to add like Laplacian transformation of the image

2. Style Encoder from the One-DM paper.
  Ethan is working on this right now and I'll be modifying my dataloader to  work with his style encoder.

Next Steps:
3. Understand ChiroDiff and implement the input preprocessing needed
4. Implement ChiroDiff model
5. Hyperparameter tuning
6. Evaluation with FID score or human ranking
