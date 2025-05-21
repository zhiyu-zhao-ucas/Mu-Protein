# Introduction 
The repository primarily hosts the code for μProtein, or Mu-Protein, uProtein, MuProtein for readability, a potent tool tailored for predicting the effects of protein mutations and navigating the fitness landscape. It is configured to facilitate the replication of the models presented in the paper titled *Accelerating protein engineering with fitness landscape modeling and reinforcement learning* which can be accessed at [this link](https://www.biorxiv.org/content/10.1101/2023.11.16.565910v5).

This repository consists of three main components:  

- **`pmlm/`** – Protein language model pretraining  
- **`mu-former/`** – Fitness landscape modeling using the pretrained protein language model  
- **`mu-search/`** – Navigating the constructed fitness landscape oracle  

For more details, refer to the respective README files:  

- [PMLM](pmlm/README.md)  
- [μFormer](mu-former/README.md)  
- [μSearch](mu-search/README.md)  

# Trademarks 
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
