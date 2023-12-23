"""
This is an example using CLAPCAP for audio captioning.
"""
from msclap import CLAP

# Load and initialize CLAP
clap_model = CLAP(version = 'clapcap', use_cuda=False)

#Load audio files
audio_files = ['audio_file']

# Generate captions for the recording
captions = clap_model.generate_caption(audio_files, resample=True, beam_size=5, entry_length=67, temperature=0.01)

# Print the result
for i in range(len(audio_files)):
    print(f"Audio file: {audio_files[i]} \n")
    print(f"Generated caption: {captions[i]} \n")

"""
The output (the exact caption may vary):

The birds are singing in the trees.
"""
