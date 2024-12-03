import os
import numpy as np
import pandas as pd
import wave

train_split_df = pd.read_csv('/mnt/sd1/jhansi/interns/chaithra/train_split_Depression_AVEC2017.csv')
test_split_df = pd.read_csv('/mnt/sd1/jhansi/interns/chaithra/dev_split_Depression_AVEC2017.csv')
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_clabel = train_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
test_split_clabel = test_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
train_split_rlabel = train_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
test_split_rlabel = test_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()

train_split_num=train_split_num[:-3]
test_split_num=test_split_num[:-3]


with open('./queries.txt') as f:
    queries = f.readlines()    

def identify_topics(sentence):
    for query in queries:
        query = query.strip('\n')
        sentence = sentence.strip('\n')
        if query == sentence:
            return True
    return False

prefix='/mnt/sd1/jhansi/interns/chaithra/data'
import wave
import librosa
import tensorflow as tf
import loupe_keras as lpk
import tensorflow_hub as hub

cluster_size=16

def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(y=signal, n_mels=80,sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    
    feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, \
                           cluster_size=cluster_size, output_dim=output_dim) \
                               (tf.convert_to_tensor(melspec))
                                
    r = feat.numpy()

    return r

elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=False)

# Function to get ELMo embeddings
def get_elmo_embeddings(x):
    x_tensor = tf.convert_to_tensor([x])
    # Extract ELMo features
    embeddings = elmo(x_tensor)
    return embeddings



def extract_features(number):
    transcript = pd.read_csv(os.path.join(prefix, '{0}_P/{0}_TRANSCRIPT.csv'.format(number)), sep='\t').fillna('')
    
    wavefile = wave.open(os.path.join(prefix, '{0}_P/{0}_AUDIO.wav'.format(number, 'r')))
    sr = wavefile.getframerate()
    nframes = wavefile.getnframes()
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
    
   
    
    response = ''
    start_time = 0
    stop_time = 0
    #feats = []
    signal = []
    txt_feats=[]
    chunk_index=1
    audio_feats, text_feats = [], []

    for t in transcript.itertuples():
        
        if getattr(t,'speaker') == 'Ellie' and (identify_topics(getattr(t,'value')) or 'i think i have asked everything' in getattr(t,'value')):
            
            if response and len(signal) > 0:
                
                # txt_feat=get_elmo_embeddings([response])
                # text_filename = f'/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/testing_data/text_feat/t_{number}_chunk_{chunk_index}.npy'
                # np.save(text_filename, txt_feat)                
                # audio_feat = wav2vlad(signal, sr)
                # audio_filename = f'/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/testing_data/audio_feat/a_{number}_chunk_{chunk_index}.npy'
                # np.save(audio_filename, audio_feat)
                audio_feats.append(wav2vlad(np.array(signal), sr))
                text_feats.append(get_elmo_embeddings(response))
                # Move to the next chunk
                chunk_index += 1

                response = ''
                signal = []
            
            
        elif getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            start_time = int(getattr(t,'start_time')*sr)
            stop_time = int(getattr(t,'stop_time')*sr)
            response += (' ' + getattr(t,'value'))
            signal = np.hstack((signal, wave_data[start_time:stop_time].astype(float)))
    
    
    print('{}_P feature done'.format(number))
    # print(len(audio_feats))
    # print(audio_feats[0].shape)
   
   # print(type(audio_feats))
    #print(audio_feats)
    #print(len(text_feats))
    return audio_feats, text_feats


''' 
# training set
for index in range(len(train_split_num)):
    extract_features(train_split_num[index])
'''

   
# for index in range(len(test_split_num)):
#     extract_features(test_split_num[index])    
    
 
def collect_and_save_features(split_num):
    
    all_audio_feats, all_text_feats, c_labels, r_labels = [], [], [], []

    for index, participant_id in enumerate(test_split_num):
        audio_feats, text_feats = extract_features(participant_id)
        
        # Convert tf.Tensor to NumPy arrays if necessary
        all_audio_feats.append(audio_feats)  # Assuming audio_feats is already a NumPy array or list of arrays
        all_text_feats.append([tf_tensor.numpy() for tf_tensor in text_feats])  # Convert each tensor to a NumPy array
    
        c_labels.append([test_split_clabel[index]])  # Classification labels per segment
        r_labels.append([test_split_rlabel[index]])
    
    #print(all_audio_feats)
    #print(all_text_feats)
    all_audio_feats = np.array(all_audio_feats, dtype=object)
    #fixed_audio_feats = [audio.squeeze(axis=1) for audio in all_audio_feats]

    all_text_feats = np.array(all_text_feats, dtype=object)
    c_labels = np.array(c_labels, dtype=object)
    r_labels = np.array(r_labels, dtype=object)

    
    # path='/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features'
    np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/new_test_audio_feats.npz', audio_features=all_audio_feats)
    # np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/test_c_labels.npz', labels=c_labels)
    # np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/test_r_labels.npz', labels=r_labels)
    # np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/test__text_feats.npz', text_features=all_text_feats)
    '''
    np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train/train_audio_feats.npz', all_audio_features)
    # np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train_c_labels.npz', labels=c_labels)
    # np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train_r_labels.npz', labels=r_labels)
    # np.savez('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train/train_text_feats.npz', text_features=all_text_feats)
    '''
    print(f"Features and labels saved  in separate files")


# Save training and test data
#collect_and_save_features(train_split_num)

collect_and_save_features(test_split_num)


