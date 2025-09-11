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

# 3. IQ Sampling
 IQ sampling is the form of sampling that an SDR performs, as well as many digital receivers (and transmitters). It’s a slightly more complex version of regular digital sampling.

 ## 3.1 Sampling Basics
 The microphone captures sound waves that are converted into electricity, and that electricity in turn is converted into numbers. The ADC acts as the bridge between the analog and digital domains. SDRs are surprisingly similar. Instead of a microphone, however, they utilize an antenna, although they also use ADCs. In both cases, the voltage level is sampled with an ADC. For SDRs, think radio waves in then numbers out.

 ****Whether we are dealing with audio or radio frequencies, we must sample if we want to capture, process, or save a signal digitally.****
**A more technical way to think of sampling a signal is grabbing values at moments in time and saving them digitally.**

## 3.2 Nyquist Sampling

The **Nyquist Rate** is the minimum rate at which a signal must be sampled to retain all of its information. According to the **Nyquist Sampling Theorem**, this rate is exactly **twice the highest frequency component** in the signal.

In other words, to accurately capture a signal, your sampling frequency ($F_s$) must be greater than or equal to twice the maximum frequency ($f_{max}$) of the signal itself ($F_s \geq 2 \cdot f_{max}$).

If a signal is sampled below this rate, a phenomenon called **aliasing** occurs. Aliasing causes high-frequency components to appear as lower frequencies in the sampled data, leading to an ambiguous and inaccurate reconstruction of the original signal. The Nyquist rate acts as a critical bridge between continuous analog signals and discrete digital signals, ensuring that this distortion is avoided.

## 3.3 Quadrature Sampling
**Quadrature Sampling** (also known as complex sampling) is the method used by a Software-Defined Radio (SDR) to sample signals.

It's a more advanced form of sampling that represents a signal's amplitude and phase using two components:

* **I (In-phase)**: The cosine component of the signal.
* **Q (Quadrature)**: The sine component, which is 90 degrees out of phase with the I component.

By adjusting the amplitudes of these two waves, you can create a single signal with any desired amplitude and phase. This approach is more practical for RF circuitry than trying to directly manipulate a single signal's amplitude and phase simultaneously. The I and Q data are then handled using **complex numbers**, which is a key concept in this type of sampling.

# 3.4 Complex Numbers
The IQ convention is an alternative way to represent magnitude and phase, which leads us to complex numbers and the ability to represent them on a complex plane.

**A complex number is really just two numbers together, a real and an imaginary portion. A complex number also has a magnitude and phase, which makes more sense if you think about it as a vector instead of a point. Magnitude is the length of the line between the origin and the point (i.e., length of the vector), while phase is the angle between the vector and 0 degrees, which we define as the positive real axis:**

![](https://pysdr.org/_images/complex_plane_1.png)

![](https://pysdr.org/_images/complex_plane_2.png)

<img width="1599" height="364" alt="image" src="https://github.com/user-attachments/assets/9c84cd6d-bdeb-420e-9657-d95a7a5e71bd" />

Based on pysdr.org, a **complex number** is a way to represent two pieces of information together: a real part and an imaginary part.

In the context of **Quadrature Sampling**, this is crucial because it allows the two signal components, **I (in-phase)** and **Q (quadrature)**, to be combined into a single number.

* The **I** data represents the **real** portion of the complex number.
* The **Q** data represents the **imaginary** portion.

When viewed as a vector on a complex plane (I on the x-axis, Q on the y-axis), a complex number has both a **magnitude** and a **phase**, which are the two key characteristics needed to fully describe a sampled signal. This representation makes it easy to analyze and manipulate signals in fields like Digital Signal Processing (DSP) and Software-Defined Radio (SDR).

## 3.5 Complex Numbers in FFTs
How **complex numbers** are used in the **Fast Fourier Transform (FFT)**.

The FFT takes a series of numbers that represent a signal over time (time-domain samples) and converts them into a representation of the signal's frequencies (frequency-domain).

The output of an FFT is a set of **complex numbers**, and each one contains three crucial pieces of information about a specific frequency in the signal:

1.  **Frequency**: This is determined by the complex number's position or index in the FFT output.
2.  **Magnitude**: This is the strength or amplitude of that frequency.
3.  **Phase**: This represents the time shift or delay of that frequency.

The magic of the FFT is that it gives you all the necessary components (magnitude and phase for each frequency) that can be added back together to perfectly reconstruct the original signal, which is where the Nyquist sampling theorem comes into play.

## 3.6 Receiver Side
How a radio receiver uses **IQ sampling** to process a signal.

### Receiver Operation

1.  A radio signal enters the receiver.
2.  Inside the receiver, the signal is split into two copies. One is the **I (in-phase)** component, and the other is the **Q (quadrature)** component, which is shifted by 90 degrees.
3.  The receiver then takes a sample from both the I and Q signals at the same time.
4.  These two samples (one I value and one Q value) are combined to create a single **complex number** in the form of $I + jQ$.

This process happens very quickly at a specific **sample rate**. The end result is that a continuous radio signal is transformed into a stream of complex numbers. This stream is what the computer-based radio (SDR) then processes to analyze the original signal.

![](https://pysdr.org/_images/IQ_diagram_rx.png)

## 3.6 Carrier and Downconversion
Explain two key concepts in radio communication: **Carrier Waves** and **Downconversion**.

### Carrier and Modulation

A **carrier wave** is a high-frequency sine wave (like an FM radio or Wi-Fi signal) used to carry information. Instead of transmitting data directly, we use the carrier as a vehicle. We add our data to the carrier by changing its properties, a process called **modulation**. For example, in an FM radio, we change the carrier's frequency to transmit sound. The information received by an SDR is stored as **I** and **Q** values, which represent the modulated carrier.

---

### Downconversion

**Downconversion** is the process of lowering the carrier wave's high frequency to a much lower, more manageable frequency (ideally 0 Hz or DC). This is done because it is extremely difficult and expensive to directly sample high-frequency signals, like a 2.4 GHz Wi-Fi signal, at the required sample rate.

To downconvert, the SDR acts like a **mixer**, multiplying the incoming high-frequency signal by a locally generated sine wave. This multiplication has the effect of shifting the signal's frequency down to a baseband, or a lower range.


By shifting the signal to a much lower frequency, an SDR can use a slower, cheaper Analog-to-Digital Converter (ADC) to sample the signal, making it practical to process radio signals on a computer.

## 3.7 Receiver Architectures

The three main radio receiver architectures.

### Direct Sampling (or Direct RF)
This is the simplest but most demanding method. The radio signal is amplified and then fed directly into an **extremely expensive ADC**. This ADC must be fast enough to sample the high-frequency radio signal as-is, without any downconversion.

***

### Direct Conversion (or Zero IF)
This is a modern approach used in many software-defined radios. It uses a mixer to immediately convert the high-frequency radio signal down to a very low frequency (0 Hz), creating the **I** and **Q** components. This process makes it possible to use slower, less expensive ADCs to capture the signal.

***

### Superheterodyne
This is a classic architecture used in older radios (like those in cars). Instead of converting the signal directly to 0 Hz, it uses a mixer to shift the signal down to a fixed **intermediate frequency (IF)**. This makes the signal easier to filter and process before it is finally converted to a digital signal by an ADC or an analog demodulator.

![](https://pysdr.org/_images/receiver_arch_diagram.svg)

## 3.8 Baseband and Bandpass Signals
The difference between **baseband** and **bandpass** signals, which are fundamental concepts in radio communication.

### Baseband Signals

A **baseband signal** is a signal centered around 0 Hz. It represents the raw data, like an audio signal or the output of a downconverted radio signal. Because it's at a low frequency, it requires a lower sample rate to capture, making it efficient to work with on a computer.

### Bandpass Signals

A **bandpass signal** is a signal that exists at some higher radio frequency (RF), away from 0 Hz. This is the type of signal that is actually transmitted through the air (e.g., Wi-Fi, FM radio). Bandpass signals are always **real** and do not contain imaginary components, as you cannot transmit imaginary data.

In summary, we work with baseband signals because they are efficient to process, but the signals we actually transmit and receive are bandpass signals. Downconversion is the process of converting a high-frequency bandpass signal into a low-frequency baseband signal for digital processing.

![](https://pysdr.org/_images/baseband_bandpass.png)

## 3.9 DC Spike and Offset Tuning

### DC Spikes

A **DC spike** is a large, unwanted spike of energy that appears in the very center of the spectrum when you use a Software-Defined Radio (SDR). It is a common problem caused by a small imperfection in the SDR hardware's oscillator, known as **LO leakage**. The spike is simply a hardware artifact and is usually not a real signal you are trying to receive. It can hide actual signals that are close to the center frequency.

### Offset Tuning

**Offset tuning** is a clever trick to avoid the DC spike. Instead of tuning the SDR directly to the frequency you want to analyze, you intentionally tune it to a nearby, "offset" frequency.

For example, to view a signal at 100 MHz, you might tune your SDR to 95 MHz instead. This moves the signal you want to see away from the center (0 Hz) where the DC spike would appear. The SDR then has to digitally shift the signal back to the correct frequency for you, which solves the problem and gives you a clean view of your desired signal.

![](https://pysdr.org/_images/dc_spike.png)
![](https://pysdr.org/_images/offtuning.png)
