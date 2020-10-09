sound_name = "agree_4" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch = c(95, 90, 90, 77, 83, 83, 113, 113, 112, 102)

s = soundgen(
  sylLen = 350,
  ampl = c(-6, -6, -20, 0, 0, 0),
  pitch = c(95, 90, 83, 115, 95, 90), 
  rolloff = c(-12, -9, -20, -10, -11, -18),
  formants = mmm,
  formantWidth = 1.5,
  noise = list(time = c(0, 150, 350, 370), value = c(-25, -25, -30, -40)),
  rolloffNoise = 0, rolloffNoiseExp = -6,
  temperature = .05, 
  addSilence = c(50, 150),
  samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
  plot = TRUE, ylim = c(0, 7)
)

seewave::savewav(s, f = 22050, filename = paste0(sound_name, '_orig.wav'))

##############
# R O B O T   V O I C E
##############
for (n in 1:nIter) {
  for (i in 1:nrow(intensPars)) {
    temp = soundgen(
      sylLen = 350 * intensPars$dur[i],
      ampl = c(-6, -6, -20, 0, 0, 0),
      pitch = c(95, 90, 83, 115, 95, 90) * robotPars$pitch, 
      rolloff = c(-12, -9, -20, -10, -11, -18) + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = list(time = c(0, 150, 350, 370) * intensPars$dur[i], 
                   value = c(-25, -25, -30, -40)),
      rolloffNoise = 0, rolloffNoiseExp = -6,
      invalidArgAction = 'ignore',
      temperature = temperature, 
      tempEffects = tempEffects,
      addSilence = c(50, 150),
      samplingRate = 22050, pitchSamplingRate = 22050, play = TRUE,
      plot = FALSE, ylim = c(0, 7)
    )
    temp_wav = tuneR::Wave(left = temp, samp.rate = 22050, bit = 16)
    temp_wav_norm = tuneR::normalize(temp_wav, level = intensPars$level[i])
    filename = paste0(sound_name, '_', intensPars$intensity[i], '_', n, '.wav')
    seewave::savewav(temp_wav_norm, f = 22050, filename = filename)
  }
}
