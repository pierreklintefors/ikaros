sound_name = "disgusted_1" 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('../00_master.R')

# pitch_manual = c(127, 116, 114, 111, 114, 115, 117, 120, 124, 126, 129, 131, 134, 132, 130, 127, 123, 117, 116, 114, 122, 130, 130, 131)

s = soundgen(
  sylLen = 660,
  ampl = c(0, 0, -10, -10),
  pitch = c(127, 111, 115, 124, 129, 132, 127, 116, 122, 131),
  rolloff = c(-20, -15, -10, -15),
  jitterDep = .25, shimmerDep = 10,
  formants = '0',
  mouth = c(.45, .45, .35, .3),
  noise = list(time = c(0, 660, 760), 
               value = c(-20, -20, -35)),
  rolloffNoise = -2,
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
      sylLen = 500 * intensPars$dur[i],
      ampl = c(0, 0, -10, -10),
      pitch = c(127, 111, 115, 124, 129, 132, 127, 116, 122, 131) * .75 * robotPars$pitch,
      rolloff = c(-20, -15, -10, -15) + intensPars$rolloff[i], 
      jitterDep = .5 * intensPars$jitter[i], 
      shimmerDep = 10 * intensPars$shimmer[i],
      formants = schwa * robotPars$formants,
      mouth = c(.45, .45, .35, .3),
      noise = list(time = c(0, 660, 760) * intensPars$dur[i], 
                   value = c(-20, -20, -35)),
      rolloffNoise = -2,
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
