sound_name = "pleased_1" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(81, 107, 123, 149, 150, 155, 154, 145, 135, 113, 104, 95, 90, 86, 80, 76, 68)

s = soundgen(
  sylLen = 640,
  attackLen = 150,
  pitch = c(81, 123, 155, 145, 113, 95, 80, 68),
  rolloff = c(-13, -13, -11, -15, -18), rolloffOct = .5,
  jitterDep = .15, shimmerDep = 5,
  formants = mmm,
  formantWidth = 1.5,
  noise = -40,
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
      attackLen = 150,
      pitch = c(81, 123, 155, 145, 113, 95, 80, 68) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-13, -13, -11, -15, -18) + intensPars$rolloff[i], 
      rolloffOct = .5,
      jitterDep = .1 * intensPars$jitter[i], 
      shimmerDep = 2 * intensPars$shimmer[i],
      formants = mmm * robotPars$formants,
      formantWidth = 1.5,
      noise = -40,
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
