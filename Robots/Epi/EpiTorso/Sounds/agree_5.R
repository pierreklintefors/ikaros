sound_name = "agree_5" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch = c(90, 77, 81, 86, 81, 99, 110, 125, 139, 145, 152)

s = soundgen(
  sylLen = 320,
  pitch = c(90, 81, 81, 110, 139, 152), 
  rolloff = c(-10, -20, -9, -12) + 3, rolloffOct = -.5,
  formants = mmm,
  formantWidth = 1.5,
  noise = list(time = c(0, 130, 320, 340), value = c(-25, -20, -30, -40)),
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
      sylLen = 320 * intensPars$dur[i],
      pitch = c(90, 81, 81, 110, 139, 152) * intensPars$pitch[i] * robotPars$pitch, 
      rolloff = c(-10, -20, -9, -12) + 3 + intensPars$rolloff[i], rolloffOct = -.5,
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = list(time = c(0, 130, 320, 340) * intensPars$dur[i], 
                   value = c(-25, -20, -30, -40)),
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

