import moviepy.editor as mpe
import librosa
import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def extract_audio_from_video(video_path, audio_path):
    video = mpe.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()

def calculate_energy(audio_samples, frame_length):
    energy = np.array([sum(abs(audio_samples[i:i+frame_length])) for i in range(0, len(audio_samples), frame_length)])
    return energy

def detect_multiple_speakers(audio_path, frame_length=2048):
    # Load audio
    audio_samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    # Calculate energy
    energy = calculate_energy(audio_samples, frame_length)
    mean_energy=np.mean(energy)
    
    # Find peaks in the energy signal
    peaks, _ = find_peaks(energy, distance=40)
    
     # Calculate mean and standard deviation of peak heights
    peak_heights = energy[peaks]
    num_peaks=len(peak_heights)
    # mean_peak_height = np.mean(peak_heights)
    
    # Filter out peaks that are too far from the mean
    filtered_peaks=[]
    for peak in peaks:
        if energy[peak] >= mean_energy:
        
            filtered_peaks.append(peak)  
            
    # Estimate number of speakers based on filtered peak count
    num_speakers = len(filtered_peaks)  # Add one to account for the possibility of no peaks
        # Check if multiple speakers are present
    if num_speakers >= num_peaks:
        msg="Multiple speakers detected."
    else:
        msg="Single speaker detected."
        
    print(num_speakers,num_peaks)   
    plot_audio_and_energy(audio_samples, energy, peaks)
    
    
    # Calculate timestamps of the detected proxies
    timestamps = [peak * frame_length / sample_rate for peak in filtered_peaks]

    return msg,num_speakers, timestamps
        
        
    # Function to plot audio signal and energy
def plot_audio_and_energy(audio_samples, energy, peaks):
    plt.figure(figsize=(12, 6))

    # Plot audio signal
    plt.subplot(2, 1, 1)
    plt.plot(audio_samples)
    plt.title('Audio Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Plot energy signal
    plt.subplot(2, 1, 2)
    plt.plot(energy)
    plt.title('Energy')
    plt.xlabel('Frame')
    plt.ylabel('Energy')

    # Plot peaks on energy signal
    plt.plot(peaks, energy[peaks], 'x', color='red', markersize=10, label='Peaks')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    video_path = "videos/vaish.mp4"
        # Output audio path
    audio_path = "vaish.wav"
    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    # Detect multiple speakers
    msg,num_speakers, timestamps=detect_multiple_speakers(audio_path)
    print(msg,num_speakers, timestamps)
