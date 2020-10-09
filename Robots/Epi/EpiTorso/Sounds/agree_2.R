sound_name = "agree_2" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch = c(142, 130, 124, 121, 118, 114, 121, 124, 133, 134, 131, 133, 137, 142, 154, 166, 181, 203, 227, 246, 259)

s = soundgen(
  sylLen = 535,
  ampl = c(0, -5, 0, 0),
  attackLen = c(70, 50),
  pitch = c(142, 118, 133, 137, 181, 259), 
  rolloff = c(-12, -10, -20, -10, -10, -12, -18),
  formants = mmm,
  formantWidth = 1.5,
  noise = c(-25, -40, -25, -40, -40, -40, -40), rolloffNoise = -12,
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
      sylLen = 535 * intensPars$dur[i],
      ampl = c(0, -5, 0, 0),
      attackLen = c(70, 50),
      pitch = c(142, 118, 133, 137, 181, 259) * .75 * intensPars$pitch[i] * robotPars$pitch, 
      rolloff = c(-12, -10, -20, -10, -10, -12, -18) + intensPars$rolloff[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = c(-25, -40, -25, -40, -40, -40, -40), rolloffNoise = -12,
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
