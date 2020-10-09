sound_name = "disgusted_4" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(98, 102, 103, 120, 134, 139, 143, 118)

s = soundgen(
  sylLen = 260,
  pitch = c(98, 103, 120, 143, 118, 110),
  rolloff = c(-14, -12),
  jitterDep = .25, shimmerDep = 10,
  formants = '0',
  formantDep = 1.2,
  mouth = c(.3, .5, .6, .5),
  noise = list(time = c(0, 260, 320),
               value = c(-25, -25, -35)),
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
      pitch = c(98, 103, 120, 143, 118, 110) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-14, -12) + intensPars$rolloff[i], 
      jitterDep = .25 * intensPars$jitter[i], 
      shimmerDep = 10 * intensPars$shimmer[i],
      formants = schwa * robotPars$formants,
      formantDep = 1.2,
      mouth = c(.3, .5, .6, .5),
      noise = list(time = c(0, 260, 320) * intensPars$dur[i],
                   value = c(-25, -25, -35)),
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
