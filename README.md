 <!-- ![Repo Logo](docs/logo.png) -->

 <p align="center">
  <img src="docs/logo-small.png" alt="Repo Logo" width="500">
</p>

# Introduction 
The repository primarily hosts the code for μProtein, or Mu-Protein, uProtein, MuProtein for readability, a potent tool tailored for predicting the effects of protein mutations and navigating the fitness landscape. It is configured to facilitate the replication of the models presented in the paper titled *Accelerating protein engineering with fitness landscape modeling and reinforcement learning* which can be accessed at [this link](https://www.biorxiv.org/content/10.1101/2023.11.16.565910v5).

This repository consists of three main components and a checkpoint folder:  

- **`pmlm/`** – Protein language model pretraining  
- **`mu-former/`** – Fitness landscape modeling using the pretrained protein language model  
- **`mu-search/`** – Navigating the constructed fitness landscape oracle
- **`pretrained/`** – The pretrained PMLM model checkpoint (LFS)

For more details, refer to the respective README files:  

- [PMLM Pretraining](pmlm/README.md)  
- [μFormer](mu-former/README.md)  
- [μSearch](mu-search/README.md)  
