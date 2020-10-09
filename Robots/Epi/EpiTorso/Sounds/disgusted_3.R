sound_name = "disgusted_3" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(120, 118, 119, 122, 127, 132, 140, 147, 153, 154, 155, 151, 148, 143, 136, 129, 125, 121, 116, 114, 118)

s = soundgen(
  sylLen = 640,
  # ampl = c(0, -10, -10),
  pitch = c(120, 122, 132, 147, 155, 143, 129, 121, 115),
  rolloff = c(-12, -10, -15) - 5, rolloffOct = 1,
  jitterDep = .25, shimmerDep = 10,
  formants = '0a',
  formantDep = 1.2,
  mouth = c(.55, .45),
  noise = -25,
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
      sylLen = 640 * intensPars$dur[i],
    ampl = c(0, -10, -10),
    pitch = c(120, 122, 132, 147, 155, 143, 129, 121, 115) * .75 * intensPars$pitch[i] * robotPars$pitch,
    rolloff = c(-12, -10, -15) - 5 + intensPars$rolloff[i], rolloffOct = 1,
    jitterDep = .25 * intensPars$jitter[i], 
    shimmerDep = 10 * intensPars$shimmer[i],
    formants = list(
      f1 = c(640, 860) * robotPars$formants,
      f2 = c(1670, 1430) * robotPars$formants,
      f3 = c(2300, 2900) * robotPars$formants,
      f4 = c(3880, 4200) * robotPars$formants
    ),
    formantDep = 1.2,
    mouth = c(.55, .45),
    noise = -25,
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
