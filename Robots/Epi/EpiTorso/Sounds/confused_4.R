sound_name = "confused_4" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(123, 124, 126, 128, 132, 133, 141, 146, 147, 162, 176, 194, 212, 232, 287, 345) 

s = soundgen(
  sylLen = 460,
  ampl = c(0, -10),
  attackLen = c(40, 100),
  pitch = c(123, 128, 133, 146, 176, 232, 345),
  rolloff = c(-17, -14),
  subDep = c(10, 25, 10), shortestEpoch = 150,
  jitterDep = .5, shimmerDep = 10,
  formants = schwaNas,
  formantDep = 1.2,
  mouth = c(.55, .45, .2),
  noise = c(-20, -25),
  rolloffNoise = -4,
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
      sylLen = 460 * intensPars$dur[i],
      ampl = c(0, -10),
      attackLen = c(40, 100),
      pitch = c(123, 128, 133, 146, 176, 232, 345) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-17, -14) + intensPars$rolloff[i], 
      subDep = c(0, 15, 5) * intensPars$sub[i], shortestEpoch = 150,
      jitterDep = .5 * intensPars$jitter[i] / 2, shimmerDep = 10 * intensPars$shimmer[i] / 2,
      formants = schwaNas,
      formantDep = 1.2,
      mouth = c(.55, .45, .2),
      noise = c(-20, -25),
      rolloffNoise = -4,
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
