sound_name = "confused_5" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(113, 105, 102, 106, 111, 123, 132, 147, 172, 193) 

s = soundgen(
  sylLen = 260,
  pitch = c(113, 102, 123, 147, 193),
  rolloff = c(-8, -10),
  formants = mmm,
  formantWidth = 1.5,
  mouth = c(.55, .4),
  noise = -25, rolloffNoise = -2,
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
      sylLen = 260 * intensPars$dur[i],
      pitch = c(113, 102, 123, 147, 193) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-8, -10) + intensPars$rolloff[i], 
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      mouth = c(.55, .4),
      noise = -25, rolloffNoise = -2,
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
