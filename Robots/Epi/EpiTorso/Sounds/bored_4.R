sound_name = "bored_4" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

s = soundgen(
  sylLen = 250,
  ampl = c(0, -10),
  pitch = c(108, 118, 108), 
  rolloff = -18,
  formants = '0',
  formantWidth = 1.5,
  mouth = c(.4, .55, .53, .5, .45),
  noise = list(time = c(-40, 50, 275, 950), 
               value = c(-30, -10, -10, -40) + 15),
  rolloffNoise = 0, rolloffNoiseExp = c(-4, -2, -3, -4, -4) - 3,
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
      sylLen = 250 * intensPars$dur[i],
      ampl = c(0, -10),
      pitch = c(108, 118, 108) * .75 * (1 / intensPars$pitch[i]) * robotPars$pitch,   # inverse rel with f0: the more bored, the lower
      rolloff = -18 + intensPars$rolloff[i], 
      formants = o * robotPars$formants,
      formantWidth = 1.5,
      mouth = c(.4, .55, .53, .5, .45),
      noise = list(time = c(-40, 50, 275, 950) * intensPars$dur[i], 
                   value = c(-30, -10, -10, -40) + 15),
      rolloffNoise = 0, rolloffNoiseExp = c(-4, -2, -3, -4, -4) - 3,
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
