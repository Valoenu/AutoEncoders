# 🎬 Movie Recommendation System with Autoencoders

As a software engineering student diving deep into AI, I built this project to explore collaborative filtering using Autoencoders in PyTorch.

> 📚 A hands-on project to understand deep learning and recommendation systems.  
> 📦 Built from scratch using the MovieLens 1M dataset.

---

## 📁 Dataset

This project uses the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/), which contains:
- **users.dat** – demographics of 6,000 users  
- **movies.dat** – 4,000+ movie titles and genres  
- **ratings.dat** – over 1 million movie ratings

Data is pre-split into:
- **Training Set:** `u1.base`  
- **Test Set:** `u1.test`

---

## 🧠 Model Overview

The model is a **Stacked Autoencoder** that learns to reconstruct user movie preferences based on their rating history.