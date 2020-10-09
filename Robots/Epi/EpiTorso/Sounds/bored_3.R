sound_name = "bored_3" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

s = soundgen(
  sylLen = 170,
  pitch = c(230, 210, 170), 
  rolloff = -20,
  formants = 'a',
  formantWidth = 1.5,
  mouth = c(.4, .55, .53, .5, .45),
  noise = list(time = c(-90, 170, 200, 330, 750, 890), 
               value = c(-10, -10, -14, -17, -30, -40) + 5),
  rolloffNoise = 0, rolloffNoiseExp = c(-4, -2, -4),
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
      sylLen = 170 * intensPars$dur[i],
      pitch = c(230, 210, 170) * .75 * (1 / intensPars$pitch[i]) * robotPars$pitch,   # inverse rel with f0: the more bored, the lower
      rolloff = -20 + intensPars$rolloff[i], 
      formants = a * robotPars$formants,
      formantWidth = 1.5,
      mouth = c(.4, .55, .53, .5, .45),
      noise = list(time = c(-90, 170, 200, 330, 750, 890) * intensPars$dur[i], 
                   value = c(-10, -10, -14, -17, -30, -40) + 5),
      rolloffNoise = 0, rolloffNoiseExp = c(-4, -2, -4),
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
