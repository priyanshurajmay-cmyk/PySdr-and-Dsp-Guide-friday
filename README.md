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


