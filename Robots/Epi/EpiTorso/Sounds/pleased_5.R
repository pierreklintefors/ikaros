sound_name = "pleased_5" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(86, 91, 94, 107, 133, 160, 202, 238, 261, 276, 292, 294, 281, 249, 200, 156, 124, 106, 97, 91, 86, 89, 90, 85, 81, 82, 77)

s = soundgen(
  sylLen = 880,
  ampl = c(0, -10),
  attackLen = 150,
  pitch = c(86, 160, 292, 124, 89, 77),
  rolloff = c(-10, -7, -12, -13) - 5,
  jitterDep = .05, shimmerDep = 2,
  formants = mmm,
  formantWidth = 2, formantDep = 1.5,
  noise = c(-25, -35), rolloffNoise = 0,
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
      sylLen = 880 * intensPars$dur[i],
      ampl = c(0, -10),
      attackLen = 150,
      pitch = c(86, 160, 292, 124, 89, 77) * intensPars$pitch[i] * robotPars$pitch,
      rolloff = c(-10, -7, -12, -13) - 5 + intensPars$rolloff[i], 
      jitterDep = .1 * intensPars$jitter[i], 
      shimmerDep = 2 * intensPars$shimmer[i],
      formants = mmm * robotPars$formants,
      formantWidth = 2, formantDep = 1.5,
      noise = c(-25, -35), rolloffNoise = 0,
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
