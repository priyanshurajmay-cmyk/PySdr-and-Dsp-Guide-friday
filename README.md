# PySdr-and-Dsp-Guide-friday

# 1. Introduction

## Software-Defined Radio (SDR):
Software-defined radio (SDR) is a radio communication system where components traditionally implemented in hardware are instead implemented in software on a computer or embedded system. This approach allows for a flexible, reconfigurable, and multi-functional device that can be adapted to different communication protocols and standards simply by changing the software.

# Digital Signal Processing (DSP):
DSP is the digital processing of signals, specifically radio frequency (RF) signals in the context of software-defined radio.
Digital signal processing (DSP) is the use of computers and specialized processors to analyze and manipulate digital signals. While our world is full of analog signals (continuous waves like sound or light), DSP works with digital signals, which are discrete values (represented as 0s and 1s). This is the core concept behind how many modern technologies, from music players to cell phones, function.

*The main objective is to gain hands-on introduction to the areas of DSP, SDR, and wireless communication*

Technologies used in the guide
1. Codes are in python.
2. They utilize numpy.
3. Matplotlib.

# 2. Frequency Domain
*This chapter introduces the frequency domain and covers Fourier series, Fourier transform, Fourier properties, FFT, windowing, and spectrograms, using Python examples.*

![](https://pysdr.org/_images/time_and_freq_domain_example_signals.png)
The frequency domain shows how a signal is made up of different frequencies. While a signal in the time domain shows how its value changes over time, a frequency domain plot shows which frequencies are present and how strong they are. The process of converting a signal from the time domain to the frequency domain is known as the Fourier Transform. This allows you to visualize the individual sine waves that make up a complex signal.

## The time domain and frequency domain are two different ways of looking at the same signal:
In the **time domain**, a signal is represented by its amplitude (strength) over a period of time. Think of it like a sound wave on an oscilloscope, where you can see the wiggles go up and down over time. It's an intuitive way to see how a signal changes instantly.

![](https://encrypted-tbn1.gstatic.com/licensed-image?q=tbn:ANd9GcSXQtCRre2ceIAR7AOEWzxhjTopoz3v8Q6KItMHn3ngo4kaYOFkRrIWnbboXzSu0FJXDIJjPV-xHS9FdtFUNHRTQ_sgxjqRN7Bm65yUlta-Jdiu_TA)

In the **frequency domain**, a signal is broken down into its individual sine wave components, and you can see which frequencies are present and how strong each one is. This is represented by a plot of amplitude vs. frequency. It's like looking at the ingredients of a complex sound—you can see exactly what pure tones (frequencies) are mixed together to create it.

### Why conversion of Time domain into Frequency domain needed
1. Converting a signal into the frequency domain allows you to see the individual frequencies that make up a complex signal. This is useful because many important signal properties, especially in communications and audio, are much easier to analyze and manipulate when viewed as their frequency components rather than as a single, complex waveform over time.
2. Converting a signal into the frequency domain shows you what frequencies are inside it. This is useful because it lets you easily filter out unwanted noise (by removing its frequency), analyze the signal's content, and compress data by getting rid of frequencies we can't perceive.
## Fourier Series
*The basics of the frequency domain start with understanding that any signal can be represented by sine waves summed together.*
**When we break a signal down into its composite sine waves, we call it a Fourier series. Here is an example of a signal that is made up of only two sine waves:**

![](https://pysdr.org/_images/summing_sinusoids.svg)

**Here , the combined reference of the time domain and the Frequency domain.**
![](https://pysdr.org/_images/fourier_series_arbitrary_function.gif)

To understand how we can break down a signal into sine waves, or sinusoids, we need to first review the three attributes of a sine wave:

1. Amplitude
2. Frequency
3. Phase

**A signal is a wave that carries information. Every wave has four key properties:**

1. Amplitude is the wave's strength or height. A taller wave means a stronger signal.

2. Frequency is how often the wave repeats. A higher frequency means more waves pass by each second.

3. Period is the amount of time it takes for one wave to complete a cycle. It's the inverse of frequency.

4. Phase is the wave's starting position. It tells you if the wave is shifted forward or backward in time.

![](https://pysdr.org/_images/amplitude_phase_period.svg)

## Time-Frequency Pairs
Time-frequency pairs refer to the relationship between a signal's properties in the time domain and its properties in the frequency domain. This is a fundamental concept in digital signal processing (DSP) and software-defined radio (SDR).

**Key Time-Frequency Pairs**
1. A single sine wave in the time domain corresponds to a single spike in the frequency domain. The sine wave's frequency will determine where the spike appears on the frequency plot.  This means a simple, repetitive signal has only one tone, or frequency.
 
![](https://pysdr.org/_images/sine-wave.png)

2. A very sharp pulse (an "impulse") in the time domain corresponds to a flat line in the frequency domain. This means a very short, sharp burst of energy contains a huge range of frequencies.

![](https://pysdr.org/_images/impulse.png)

3. The frequency domain has a strong spike, which happens to be at the frequency of the square wave, but there are more spikes as we go higher in frequency. It is due to the quick change in time domain, just like in the previous example. But it’s not flat in frequency. It has spikes at intervals, and the level slowly decays (although it will continue forever). A square wave in time domain has a sin(x)/x pattern in the frequency domain (a.k.a. the sinc function).

![](https://pysdr.org/_images/square-wave.svg)

4. A constant, unchanging signal (a DC signal) in the time domain corresponds to a single spike at 0 Hz in the frequency domain. Since the signal is not changing at all, it has no frequency other than zero.

![](https://pysdr.org/_images/dc-signal.png)

## Fourier Transform

The Fourier Transform is a mathematical tool that converts a signal from the time domain to the frequency domain and back.

1. The first equation is the Continuous Fourier Transform, which takes a signal over time, x(t), and gives you its frequency content, X(f). It's used for theoretical math problems.

2. The third equation is the Discrete Fourier Transform (DFT), the digital version used in code. It replaces the continuous integral with a summation over a set of samples, which is what computers use.


## Time-Frequency Properties
Time-frequency properties are fundamental relationships between how a signal behaves in the time domain and its corresponding content in the frequency domain. These properties are critical for understanding and manipulating signals in digital signal processing (DSP) and software-defined radio (SDR).

**1. Linearity Property:** If we add two signals in time, then the frequency domain version will also be the two frequency domain signals added together. It also tells us that if we multiply either one by a scaling factor, the frequency domain will also scale by the same amount.

**2. Frequency Shift Property:** 

<img width="334" height="84" alt="image" src="https://github.com/user-attachments/assets/26364231-64d3-4489-a9ca-1d8c9f40dee6" />

This property tells us that if we take a signal x(t) and multiply it by a sine wave, then in the frequency domain we get x(f) except shifted by a certain frequency, f0. This shift in frequency may be easier to visualize:
![](https://pysdr.org/_images/freq-shift.svg)

Frequency shift is integral to DSP because we will want to shift signals up and down in frequency for many reasons. This property tells us how to do that (multiply by a sine wave). Here’s another way to visualize this property:

![](https://pysdr.org/_images/freq-shift-diagram.svg)

**3. Scaling in Time Property:**

<img width="310" height="106" alt="image" src="https://github.com/user-attachments/assets/9a0bfa63-e455-4b18-aa03-dc906def74d0" />

Scaling in time essentially shrinks or expands the signal in the x-axis. What this property tells us is that scaling in the time domain causes inverse scaling in the frequency domain. For example, when we transmit bits faster we have to use more bandwidth. The property helps to explain why higher data rate signals take up more bandwidth/spectrum. If time-frequency scaling was proportional instead of inversely proportional, cellular carriers would be able to transmit all the bits per second they wanted without paying billions for spectrum!

<img width="1555" height="341" alt="image" src="https://github.com/user-attachments/assets/1f3f08d9-a3e1-4a26-b06d-251adc5f93be" />


**4. Convolution in Time Property:** When we convolve time domain signals, it’s equivalent to multiplying the frequency domain versions of those two signals. It is very different from adding together two signals.

![](https://pysdr.org/_images/two-signals.svg)

**5. Convolution in Frequency Property:** The convolution property works in reverse

## Fast Fourier Transform (FFT)
The Fast Fourier Transform (FFT) is simply an algorithm to compute the discrete Fourier Transform.
**The FFT is a function with one input and one output. It converts a signal from time to frequency:**

<img width="732" height="131" alt="image" src="https://github.com/user-attachments/assets/342bbf54-ef2d-4337-a392-0f53697fd0d6" />

We will only be dealing with 1 dimension FFTs in this textbook (2D is used for image processing and other applications). For our purposes, think of the FFT function as having one input: a vector of samples, and one output: the frequency domain version of that vector of samples. **The size of the output is always the same as the size of the input. If I feed 1,024 samples into the FFT, I will get 1,024 out. The confusing part is that the output will always be in the frequency domain, and thus the “span” of the x-axis if we were to plot it doesn’t change based on the number of samples in the time domain input.** Let’s visualize that by looking at the input and output arrays, along with the units of their indices:

<img width="1619" height="323" alt="image" src="https://github.com/user-attachments/assets/c0dc2c67-d26f-48c6-9ad2-104a66026744" />

**Because the output is in the frequency domain, the span of the x-axis is based on the sample rate.**
**When we use more samples for the input vector, we get a better resolution in the frequency domain (in addition to processing more samples at once).**

## Negative Frequencies
There isn’t really such thing as a “negative frequency” when it comes to transmitting/receiving RF signals, it’s just a representation we use. Here’s an intuitive way to think about it. Consider we tell our SDR to tune to 100 MHz (the FM radio band) and sample at a rate of 10 MHz. In other words, we will view the spectrum from 95 MHz to 105 MHz. Perhaps there are three signals present:


<img width="900" height="298" alt="image" src="https://github.com/user-attachments/assets/53084b6b-0c1c-4f81-aa0f-b1b76b1cf7dd" />


<img width="1007" height="375" alt="image" src="https://github.com/user-attachments/assets/f3f809b8-aed3-435b-888d-8ce509b8d15c" />

**We tuned the SDR to 100 MHz. So the signal that was at about 97.5 MHz shows up at -2.5 MHz when we represent it digitally, which is technically a negative frequency. In reality it’s just a frequency lower than the center frequency.**

## Order in Time Doesn’t Matter
Changing the order things happen in the time domain doesn’t change the frequency components in the signal. I.e., doing a single FFT of the following two signals will both have the same two spikes because the signal is just two sine waves at different frequencies. Changing the order the sine waves occur doesn’t change the fact that they are two sine waves at different frequencies. This assumes both sine waves occur within the same time span fed into the FFT.


<img width="1008" height="361" alt="image" src="https://github.com/user-attachments/assets/6f8b20f6-8919-4c69-a0aa-c4784c77812e" />

## FFT in Python
**OBJECTIVE**:let’s actually look at some Python code and use Numpy’s FFT function, np.fft.fft().

<img width="800" height="1219" alt="image" src="https://github.com/user-attachments/assets/b675dfe2-b730-4859-8244-5a905506ecc4" />

<img width="1402" height="507" alt="image" src="https://github.com/user-attachments/assets/ae237866-3edd-43d2-84d0-d19003f96e78" />

Next let’s use NumPy’s FFT function:
```S = np.fft.fft(s)```

<img width="404" height="178" alt="image" src="https://github.com/user-attachments/assets/ee16a948-6594-489a-af68-6627405fa93f" />
<img width="701" height="192" alt="image" src="https://github.com/user-attachments/assets/0039f202-6050-42e9-b332-72d81731ed57" />

**Got the array of S in complex numbers, as mentioned in the guide**


<img width="1425" height="607" alt="image" src="https://github.com/user-attachments/assets/b3703b86-5c1f-4b77-8e89-cc16f76c9950" />

**Also got the required radians and values in terminal**

### FFT shift using python

```S = np.fft.fftshift(np.fft.fft(s))```

**Final Code Example**

```
import numpy as np
import matplotlib.pyplot as plt

Fs = 1 # Hz
N = 100 # number of points to simulate, and our FFT size

t = np.arange(N) # because our sample rate is 1 Hz
s = np.sin(0.15*2*np.pi*t)
S = np.fft.fftshift(np.fft.fft(s))
S_mag = np.abs(S)
S_phase = np.angle(S)
f = np.arange(Fs/-2, Fs/2, Fs/N)
plt.figure(0)
plt.plot(f, S_mag,'.-')
plt.figure(1)
plt.plot(f, S_phase,'.-')
plt.show()
```
Got the Code running and the required plots

<img width="2156" height="1375" alt="image" src="https://github.com/user-attachments/assets/1cd2dcef-388e-4d7a-ac63-203dac3883e6" />

## Windowing and why its needed
When we use an FFT to measure the frequency components of our signal, the FFT assumes that it’s being given a piece of a periodic signal. It behaves as if the piece of signal we provided continues to repeat indefinitely. It’s as if the last sample of the slice connects back to the first sample. It stems from the theory behind the Fourier Transform. It means that we want to avoid sudden transitions between the first and last sample because sudden transitions in the time domain look like many frequencies, and in reality our last sample doesn’t actually connect back to our first sample. To put it simply: if we are doing an FFT of 100 samples, using np.fft.fft(x), we want x[0] and x[99] to be equal or close in value.

<img width="1288" height="586" alt="image" src="https://github.com/user-attachments/assets/1a26b0e3-03c2-4d18-aceb-5c1a0592b51e" />

A basic window in python for previous example is ```s = s * np.hamming(100)```

## FFT Sizing
The number of data samples used to perform a single Fast Fourier Transform (FFT) calculation. This process converts a signal from the time domain to the frequency domain, which is essential for creating a visual spectrogram.

**The Key Trade-off: Time vs. Frequency
The size of the FFT is a critical trade-off between two types of resolution:**

1. Frequency Resolution: A larger FFT size provides better frequency resolution, allowing you to distinguish between very close frequencies. Think of this as zooming in on the frequency spectrum to see fine details.

2. Time Resolution: A smaller FFT size provides better time resolution, enabling you to see rapid, quick changes in the signal. This is like a fast camera shutter, capturing quick events.

Essentially, you can't have both at the same time. A large FFT size gives you a detailed look at frequencies but blurs fast changes, while a small FFT size captures quick changes but with less detail in the frequency spectrum. The choice of FFT size depends on whether you're looking for static, precise frequencies or for dynamic, fleeting events.

## Spectrogram/Waterfall
A spectrogram is the plot that shows frequency over time. It is simply a bunch of FFTs stacked together (vertically, if you want frequency on the horizontal axis).

![](https://pysdr.org/_images/spectrogram_diagram.svg)

**Got the required spectrogram after running the example code**
```
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6

# Generate tone plus noise
t = np.arange(1024*1000)/sample_rate # time vector
f = 50e3 # freq of tone
x = np.sin(2*np.pi*f*t) + 0.2*np.random.randn(len(t))
# simulate the signal above, or use your own signal

fft_size = 1024
num_rows = len(x) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(x)/sample_rate, 0])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.show()
```

<img width="1635" height="1278" alt="image" src="https://github.com/user-attachments/assets/9240c646-5bac-4323-aafd-f3b45efc42b2" />

## FFT Implementation using Python

The code down here shows how FFT runs under the hood, also mentioned as Cooley-Tukey FFT algorithm.

```
import numpy as np
import matplotlib.pyplot as plt

def fft(x):
    N = len(x)
    if N == 1:
        return x
    twiddle_factors = np.exp(-2j * np.pi * np.arange(N//2) / N)
    x_even = fft(x[::2]) # yay recursion!
    x_odd = fft(x[1::2])
    return np.concatenate([x_even + twiddle_factors * x_odd,
                           x_even - twiddle_factors * x_odd])

# Simulate a tone + noise
sample_rate = 1e6
f_offset = 0.2e6 # 200 kHz offset from carrier
N = 1024
t = np.arange(N)/sample_rate
s = np.exp(2j*np.pi*f_offset*t)
n = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2) # unity complex noise
r = s + n # 0 dB SNR

# Perform fft, fftshift, convert to dB
X = fft(r)
X_shifted = np.roll(X, N//2) # equivalent to np.fft.fftshift
X_mag = 10*np.log10(np.abs(X_shifted)**2)

# Plot results
f = np.linspace(sample_rate/-2, sample_rate/2, N)/1e6 # plt in MHz
plt.plot(f, X_mag)
plt.plot(f[np.argmax(X_mag)], np.max(X_mag), 'rx') # show max
plt.grid()
plt.xlabel('Frequency [MHz]')
plt.ylabel('Magnitude [dB]')
plt.show()
```

**FFT implementation example code with a successful plot output.**

<img width="1586" height="1293" alt="image" src="https://github.com/user-attachments/assets/70d2a63a-7120-4cd8-bb26-fe3c29c4b748" />

