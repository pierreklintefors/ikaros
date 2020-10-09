sound_name = "pleased_2" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(95, 97, 100, 103, 115, 130, 152, 179, 204, 219, 233, 230, 214, 198, 181, 162, 144, 129, 124, 118, 111, 105, 102, 98, 94, 92, 86, 88, 87, 86)

s = soundgen(
  sylLen = 970,
  attackLen = 150,
  pitch = c(95, 130, 233, 162, 118, 94, 86),
  rolloff = c(-15, -14, -11, -16, -18),
  jitterDep = 0, shimmerDep = 5,
  formants = mmm,
  formantWidth = 1.2,
  mouth = c(.6, .4),
  noise = c(-40, -30, -40), rolloffNoise = -4,
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
      sylLen = 970 / 1.25 * intensPars$dur[i],
      attackLen = 150,
      pitch = c(95, 130, 233, 162, 118, 94, 86) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-15, -14, -11, -17, -22) + intensPars$rolloff[i], 
      rolloffOct = .5,
      jitterDep = .2 * intensPars$jitter[i], 
      shimmerDep = 2 * intensPars$shimmer[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.2,
      mouth = c(.6, .4),
      noise = c(-40, -30, -40), rolloffNoise = -4,
      invalidArgAction = 'ignore',
      temperature = temperature / 2, 
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
