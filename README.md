# Depression_detection

Dataset used: DAIC-WOZ

CODE :

Classification

- preprocess.py: extract audio and text features

- audio_gru.py: train audio network
  
- text_bilstm.py: train text network
  
- fuse_net_whole.py: train fuse network

Feature Extraction:

In preprocess.py
- get_elmo_embeddings--> for getting elmo embeddings from text
  
- wav2vlad--> for getting processed netvlad embeddings from audio
  
- extract_features---> calls both the above functions
