# Depression_detection

Dataset used: DAIC-WOZ

CODE :

Classification

- preprocess: extract audio and text features

- audio_gru.ipynb: train audio network
  
- text_bilstm..ipynb: train text network
  
- fusion.py: train fuse network

Feature Extraction:

In preprocess.py
- get_elmo_embeddings--> for getting elmo embeddings from text
  
- wav2vlad--> for getting processed netvlad embeddings from audio
  
- extract_features---> calls both the above functions
