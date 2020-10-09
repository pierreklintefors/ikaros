sound_name = "confused_3" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(115, 116, 113, 110, 110, 112, 117, 130, 143, 162, 188, 208, 241, 276, 304, 342) 

s = soundgen(
  sylLen = 420,
  ampl = c(0, -15),
  pitch = c(115, 110, 112, 130, 188, 276, 342),
  rolloff = c(-19, -16),
  formants = 'a',
  formantWidth = 1.5,
  mouth = c(.55, .5, .55),
  noise = list(time= c(-50, 420),
               value = c(-20, -25)),
  rolloffNoise = -5,
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
      sylLen = 420 * intensPars$dur[i],
      ampl = c(0, -15),
      pitch = c(115, 110, 112, 130, 188, 276, 342) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-19, -16) + intensPars$rolloff[i], 
      formants = a * robotPars$formants,
      formantWidth = 1.5,
      mouth = c(.55, .5, .55),
      noise = list(time= c(-50, 420) * intensPars$dur[i],
                   value = c(-20, -25)),
      rolloffNoise = -5,
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
