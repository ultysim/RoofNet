# RoofNet
Deep learning project to identify roof age using historical satellite images to lower the customer acquisition cost for new solar installations. 

Project in conjuction with Misha Laskin, Chris Heinrich, and Evert P.L. van Nieuwenburg

Paper available: https://arxiv.org/abs/2001.04227

# Motivation

Rooftop solar is one of the most promising tools for drawing down greenhouse gas (GHG) emissions and is cost-competitive with fossil fuels in many areas of the world today. One of the most important criteria for determining the suitability of a building for rooftop solar is the current age of its roof. The reason for this is simple -- because rooftop solar installations are long-lived, the roof needs to be new enough to last for the lifetime of the solar array or old enough to justify being replaced. We present a data-driven method for determining the age of a roof from historical satellite imagery, which removes one of the last obstacles to a fully automated pipeline for rooftop solar site selection.

# Approach 
Our model is comprised of a convolutional variational autoencoder and deep fully connected binary classifier trained using historic satellite rooftop images. True labels were scraped from public reroof permit data.

At inference time images are encoded into latent vectors by the VAE and concatenated for classification. Roof age is determined as the argmax over temporally adjacent images. A roof age is undefined if no pair of images classify as a reroof.

# Algorithms
The VAE model is located in the source directory.

The notebook directory contains a script to preprocess the images and relevant metadata as well as the experimental notebook with results. 

TODO:
Move binary classifier from notebook to source.
